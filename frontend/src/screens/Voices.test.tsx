import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { CloudSlotsOut, EngineVoicesOut, VoiceOut } from "../api/types";
import { makeJob } from "../test/fixtures";
import { errorResponse, jsonResponse, mockApi, renderWithProviders } from "../test/utils";
import { Voices } from "./Voices";

function makeVoice(overrides: Partial<VoiceOut> = {}): VoiceOut {
  return {
    voice_id: "v1",
    name: "Narrator",
    kind: "preset",
    engine: "kokoro",
    preset_id: "af_heart",
    blend: null,
    reference_audio: null,
    seed: 1234,
    consent_attested: false,
    consent: null,
    tags: [],
    created_at: "2026-07-01T00:00:00Z",
    has_audition: false,
    ...overrides,
  };
}

const kokoroCatalog: EngineVoicesOut = {
  engine_id: "kokoro",
  voices: [
    { id: "af_heart", name: "Heart", language: "en-US", gender: "female", description: null },
    { id: "af_nicole", name: "Nicole", language: "en-US", gender: "female", description: null },
    { id: "bf_emma", name: "Emma", language: "en-GB", gender: "female", description: null },
  ],
};

/** Register the three GETs the screen mounts with (jobs poll is pre-registered). */
function mountServer({
  voices = [] as VoiceOut[],
  unreadable = [] as { voice_id: string; error: string }[],
  slots = { max_slots: 10, count: 0, slots: [] } as CloudSlotsOut,
} = {}) {
  return mockApi()
    .get("/api/voices", { voices, unreadable })
    .get("/api/cloud-slots", slots)
    .get("/api/books", { books: [] });
}

describe("Voices", () => {
  it("renders the library from /api/voices with consent state, slot bank, and unreadable warnings", async () => {
    mountServer({
      voices: [
        makeVoice({ voice_id: "vp", name: "Bright Narrator", seed: 42 }),
        makeVoice({
          voice_id: "vc",
          name: "Mr. Darcy",
          kind: "cloned",
          engine: "chatterbox",
          preset_id: null,
          reference_audio: "reference.wav",
          consent_attested: true,
          consent: { attested_by: "Cy", reference_sha256: "abc", attested_at: "2026-07-02T10:00:00Z" },
        }),
      ],
      unreadable: [{ voice_id: "vbad", error: "meta.json corrupt" }],
      slots: { max_slots: 10, count: 3, slots: [] },
    });
    renderWithProviders(<Voices />);

    expect(await screen.findByText("Bright Narrator")).toBeInTheDocument();
    expect(screen.getByText("Mr. Darcy")).toBeInTheDocument();
    expect(screen.getByText("2 in library")).toBeInTheDocument();
    expect(screen.getByText(/cloud slots 3\/10/)).toBeInTheDocument();
    expect(screen.getByText("af_heart · seed 42")).toBeInTheDocument();
    expect(screen.getByText(/consent: attested · 2026-07-02/)).toBeInTheDocument();
    expect(screen.getByText(/vbad: meta.json corrupt/)).toBeInTheDocument();
  });

  it("creating a kokoro preset voice POSTs the recipe and closes the dialog", async () => {
    const user = userEvent.setup();
    const server = mountServer()
      .get("/api/engines/kokoro/voices", kokoroCatalog)
      .post("/api/voices", makeVoice({ voice_id: "new", name: "Bright Narrator" }));
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: "add voice" }));
    const dialog = await screen.findByRole("dialog", { name: "Add voice" });
    const submit = within(dialog).getByRole("button", { name: "add voice" });
    expect(submit).toBeDisabled(); // a voice needs a name before it can exist

    await user.type(within(dialog).getByPlaceholderText("Narrator"), "Bright Narrator");
    expect(submit).toBeEnabled();
    await user.click(submit);

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.jsonBodyOf("POST", "/api/voices")).toEqual({
      kind: "preset",
      name: "Bright Narrator",
      engine: "kokoro",
      preset_id: "af_heart",
    });
  });

  it("manual blend POSTs the mixer's layers and blocks a cross-accent mix at the fader", async () => {
    const user = userEvent.setup();
    const server = mountServer()
      .get("/api/engines/kokoro/voices", kokoroCatalog)
      .post("/api/voices", makeVoice({ voice_id: "blend1", kind: "blend" }));
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: "add voice" }));
    const dialog = await screen.findByRole("dialog", { name: "Add voice" });
    await user.click(within(dialog).getByRole("button", { name: "blend" }));
    await user.click(within(dialog).getByRole("button", { name: "manual mix" }));
    await user.type(within(dialog).getByPlaceholderText("Narrator"), "Duet");

    // an American + British layer mix is invalid — the dialog refuses before the server would
    await user.click(within(dialog).getByRole("button", { name: /layer 1 preset/ }));
    await user.click(await screen.findByRole("option", { name: /bf_emma/ }));
    expect(await within(dialog).findByText(/can't blend across accents/)).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "add voice" })).toBeDisabled();

    await user.click(within(dialog).getByRole("button", { name: /layer 1 preset/ }));
    await user.click(await screen.findByRole("option", { name: /af_heart/ }));
    await waitFor(() =>
      expect(within(dialog).queryByText(/can't blend across accents/)).not.toBeInTheDocument(),
    );

    await user.click(within(dialog).getByRole("button", { name: "add voice" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.jsonBodyOf("POST", "/api/voices")).toEqual({
      kind: "blend",
      name: "Duet",
      components: [
        { preset_id: "af_heart", weight: 60 },
        { preset_id: "af_nicole", weight: 40 },
      ],
    });
  });

  it("clone uploads multipart with consent fields and escalates reclone_blocked into replace", async () => {
    const user = userEvent.setup();
    const server = mountServer();
    server.on("POST", "/api/voices/clone", (_url, init) => {
      const form = init?.body as FormData;
      return form.get("replace") === "true"
        ? jsonResponse(makeVoice({ voice_id: "vc", kind: "cloned", engine: "indextts2" }))
        : errorResponse(409, "reclone_blocked", "this reference clip is already cloned");
    });
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: "new clone" }));
    const dialog = await screen.findByRole("dialog", { name: "New cloned voice" });
    const submit = within(dialog).getByRole("button", { name: "clone voice" });
    expect(submit).toBeDisabled();

    const fileInput = dialog.querySelector<HTMLInputElement>('input[type="file"]');
    expect(fileInput).not.toBeNull();
    await user.upload(fileInput!, new File(["wav bytes"], "darcy-ref.wav", { type: "audio/wav" }));
    await user.type(within(dialog).getByPlaceholderText("Mr. Darcy"), "Mr. Darcy");
    expect(submit).toBeDisabled(); // no consent attestation yet — the gate holds

    await user.click(within(dialog).getByRole("checkbox", { name: /permission to clone/ }));
    await user.type(within(dialog).getByPlaceholderText("your name"), "Cyber");
    expect(submit).toBeEnabled();

    // the engine picker offers all three clone targets; pick the non-default local one
    await user.click(within(dialog).getByRole("button", { name: /chatterbox — local, free/ }));
    expect(await screen.findByRole("option", { name: /^chatterbox/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /^elevenlabs/ })).toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: /^indextts2/ }));

    await user.click(submit);
    expect(await within(dialog).findByText("reclone_blocked")).toBeInTheDocument();
    const form = server.formBodyOf("POST", "/api/voices/clone");
    expect((form.get("file") as File).name).toBe("darcy-ref.wav");
    expect(form.get("name")).toBe("Mr. Darcy");
    expect(form.get("engine")).toBe("indextts2");
    expect(form.get("consent")).toBe("true");
    expect(form.get("attested_by")).toBe("Cyber");
    expect(form.get("replace")).toBeNull();

    await user.click(within(dialog).getByRole("button", { name: /replace it/ }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.formBodyOf("POST", "/api/voices/clone").get("replace")).toBe("true");
  });

  it("audition on a paid engine 402s, then re-sends confirm_paid=true from the confirm key", async () => {
    const user = userEvent.setup();
    const server = mountServer({
      voices: [makeVoice({ voice_id: "v11", name: "Cloud Belle", engine: "elevenlabs", preset_id: "stock1" })],
    });
    server.on("POST", "/api/voices/v11/audition", (_url, init) => {
      const body = JSON.parse(init?.body as string) as { confirm_paid: boolean };
      return body.confirm_paid
        ? jsonResponse({ voice_id: "v11", duration_seconds: 3.1, cost_usd: 0.0123, audition_url: "/x" })
        : errorResponse(402, "payment_confirmation_required", "this audition bills elevenlabs", {
            estimated_usd: 0.0123,
          });
    });
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: /^audition/ }));
    expect(await screen.findByText("payment_confirmation_required")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /confirm ~\$0\.0123 & play/ }));

    await waitFor(() =>
      expect(screen.queryByText("payment_confirmation_required")).not.toBeInTheDocument(),
    );
    const posts = server.calls.filter((c) => c.method === "POST" && c.url.includes("/audition"));
    expect(posts).toHaveLength(2);
    expect(JSON.parse(posts[0].body as string)).toEqual({ confirm_paid: false });
    expect(JSON.parse(posts[1].body as string)).toEqual({ confirm_paid: true });
  });

  it("engine_cold refusal offers warm-up: POSTs the voice's engine warmup and resets to the audition control", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "k1", name: "Cold Heart" })] });
    server.error("POST", "/api/voices/k1/audition", 409, "engine_cold", "kokoro has no warm worker");
    server.post("/api/engines/kokoro/warmup", makeJob({ job_id: "w1", kind: "warmup", state: "queued" }));
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: /^audition/ }));
    expect(await screen.findByText("engine_cold")).toBeInTheDocument();
    expect(screen.getByText(/kokoro has no warm worker/)).toBeInTheDocument();

    // the recourse targets the VOICE's engine, and success clears back to an idle audition
    await user.click(screen.getByRole("button", { name: "warm up first" }));
    await waitFor(() => expect(server.lastCall("POST", "/warmup")?.url).toBe("/api/engines/kokoro/warmup"));
    expect(await screen.findByRole("button", { name: /^audition/ })).toBeInTheDocument();
    expect(screen.queryByText("engine_cold")).not.toBeInTheDocument();
  });

  // Fake timers drive the backoff deterministically. fireEvent (not userEvent) on purpose:
  // RTL's asyncWrapper awaits a setTimeout(0) it only advances under a `jest` global, so
  // userEvent deadlocks under vitest fake timers. Timer advances are act()-wrapped instead.
  it("gpu_busy_retry auto-retries keep confirm_paid=true, stop at the bound, and the manual retry stays paid", async () => {
    vi.useFakeTimers();
    try {
      const bodies: { confirm_paid: boolean }[] = [];
      mountServer({
        voices: [makeVoice({ voice_id: "v11", name: "Cloud Belle", engine: "elevenlabs", preset_id: "stock1" })],
      }).on("POST", "/api/voices/v11/audition", (_url, init) => {
        const body = JSON.parse(init?.body as string) as { confirm_paid: boolean };
        bodies.push(body);
        return body.confirm_paid
          ? errorResponse(402, "gpu_busy_retry", "a render is lending the GPU between segments")
          : errorResponse(402, "payment_confirmation_required", "this audition bills elevenlabs", {
              estimated_usd: 0.0123,
            });
      });
      renderWithProviders(<Voices />);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50); // settle the mount queries
      });

      fireEvent.click(screen.getByRole("button", { name: /^audition/ }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });
      expect(screen.getByText("payment_confirmation_required")).toBeInTheDocument();
      expect(bodies).toEqual([{ confirm_paid: false }]);

      // the user confirms the paid audition; the server then refuses softly (GPU borrowed)
      fireEvent.click(screen.getByRole("button", { name: /confirm ~\$0\.0123 & play/ }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });
      expect(bodies).toEqual([{ confirm_paid: false }, { confirm_paid: true }]);
      expect(screen.getByText(/queued behind a render segment — retrying/)).toBeInTheDocument();

      // bounded auto-retry with backoff: every re-POST must keep the user's confirm_paid=true
      for (let i = 0; i < 6; i++) {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(2000);
        });
      }
      expect(bodies).toHaveLength(5); // 1 unpaid + 1 explicit confirm + exactly 3 auto-retries
      expect(bodies.slice(1)).toEqual(Array(4).fill({ confirm_paid: true }));
      expect(screen.getByText(/the render keeps running/)).toBeInTheDocument();

      // ...and the budget is truly exhausted: no further auto-POST however long the clock runs
      await act(async () => {
        await vi.advanceTimersByTimeAsync(60_000);
      });
      expect(bodies).toHaveLength(5);

      // the manual retry also preserves the paid confirmation — never a fresh unpaid attempt
      fireEvent.click(screen.getByRole("button", { name: "retry" }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });
      expect(bodies).toHaveLength(6);
      expect(bodies[5]).toEqual({ confirm_paid: true });
    } finally {
      vi.useRealTimers();
    }
  });

  it("audition on a local engine fires once with confirm_paid=false and plays the take", async () => {
    const user = userEvent.setup();
    const server = mountServer({
      voices: [makeVoice({ voice_id: "k1", name: "Local Heart", has_audition: true })],
    }).post("/api/voices/k1/audition", {
      voice_id: "k1",
      duration_seconds: 2.5,
      cost_usd: 0,
      audition_url: "/api/voices/k1/audition.wav",
    });
    const { container } = renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: /^audition/ }));

    await waitFor(() => {
      const audio = container.querySelector("audio");
      expect(audio).not.toBeNull();
      expect(audio!.src).toContain("/api/voices/k1/audition.wav");
    });
    const posts = server.calls.filter((c) => c.method === "POST" && c.url.includes("/audition"));
    expect(posts).toHaveLength(1);
    expect(JSON.parse(posts[0].body as string)).toEqual({ confirm_paid: false });
    expect(screen.queryByText("payment_confirmation_required")).not.toBeInTheDocument();
  });

  it("delete refuses with voice_referenced and never DELETEs while books still cast the voice", async () => {
    const user = userEvent.setup();
    const server = mountServer({
      voices: [makeVoice({ voice_id: "v1", name: "Busy Voice" })],
    }).get("/api/voices/v1/references", {
      voice_id: "v1",
      references: [{ book_id: "b1", role: "narrator" }],
    });
    renderWithProviders(<Voices />);

    await screen.findByText("Busy Voice");
    await user.click(screen.getByRole("button", { name: "✕" }));

    expect(await screen.findByText("voice_referenced")).toBeInTheDocument();
    expect(screen.getByText(/still assigned in: b1 \(narrator\)/)).toBeInTheDocument();
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(0);
    expect(screen.getByText("Busy Voice")).toBeInTheDocument();
  });

  it("delete proceeds when the references check comes back clear", async () => {
    const user = userEvent.setup();
    const server = mountServer({
      voices: [makeVoice({ voice_id: "v1", name: "Solo Voice" })],
    })
      .get("/api/voices/v1/references", { voice_id: "v1", references: [] })
      .delete("/api/voices/v1", { deleted: "v1" });
    renderWithProviders(<Voices />);

    await screen.findByText("Solo Voice");
    server.get("/api/voices", { voices: [], unreadable: [] }); // the post-delete refetch finds an empty booth
    await user.click(screen.getByRole("button", { name: "✕" }));

    await waitFor(() => expect(screen.queryByText("Solo Voice")).not.toBeInTheDocument());
    expect(server.lastCall("DELETE", "/api/voices/v1")).toBeDefined();
    expect(await screen.findByText(/no voices yet/)).toBeInTheDocument();
  });

  it("saving edited tags PATCHes the parsed tag list", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", tags: [] })] });
    server.on("PATCH", "/api/voices/v1", () => jsonResponse(makeVoice({ tags: ["hero", "gruff"] })));
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: "+ tag" }));
    await user.type(screen.getByPlaceholderText("comma, separated, tags"), "hero, gruff");
    await user.click(screen.getByRole("button", { name: "save" }));

    await waitFor(() =>
      expect(screen.queryByPlaceholderText("comma, separated, tags")).not.toBeInTheDocument(),
    );
    expect(server.jsonBodyOf("PATCH", "/api/voices/v1")).toEqual({ tags: ["hero", "gruff"] });
  });

  it("the mixer demo plays through a blob URL and revokes it when the dialog closes (leak guard)", async () => {
    const user = userEvent.setup();
    const createSpy = vi.spyOn(URL, "createObjectURL");
    const revokeSpy = vi.spyOn(URL, "revokeObjectURL");
    const server = mountServer().get("/api/engines/kokoro/voices", kokoroCatalog);
    server.on("GET", "/api/engines/kokoro/preview", () =>
      new Response(new Blob(["RIFFfake"], { type: "audio/wav" }), { status: 200 }),
    );
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: "add voice" }));
    const dialog = await screen.findByRole("dialog", { name: "Add voice" });
    await user.click(within(dialog).getByRole("button", { name: "▶ demo" }));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(server.lastCall("GET", "/api/engines/kokoro/preview")?.url).toBe(
      "/api/engines/kokoro/preview?preset=af_heart",
    );
    expect(await within(dialog).findByRole("button", { name: "playing…" })).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    const objectUrl = createSpy.mock.results[0].value as string;
    expect(revokeSpy).toHaveBeenCalledWith(objectUrl);
    createSpy.mockRestore();
    revokeSpy.mockRestore();
  });
});
