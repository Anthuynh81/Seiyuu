import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { CharactersOverview, CloudSlotsOut, EngineVoicesOut, VoiceOut } from "../api/types";
import { makeBook, makeJob } from "../test/fixtures";
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

/** A minimal /characters payload: just the id -> name pairs the delete dialog reads. */
function charactersOf(bookId: string, chars: { id: string; name: string }[]): CharactersOverview {
  return {
    book_id: bookId,
    provider_id: "local",
    model_id: "m",
    prompt_version: "v1",
    narration_segments: 0,
    low_confidence_segments: 0,
    unattributed_quote_segments: 0,
    confidence_threshold: 0.5,
    characters: chars.map((c) => ({
      ...c,
      aliases: [],
      gender: null,
      age_hint: null,
      line_count: 1,
      sample_lines: [],
      first_appearance: null,
    })),
    flagged: [],
    notes: [],
    edit_warnings: [],
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

  it("rename PATCHes name only and shows the new name without touching the recipe", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Narrator", tags: ["hero"] })] });
    server.on("PATCH", "/api/voices/v1", (_url, init) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse(makeVoice({ voice_id: "v1", name: String(body.name), tags: ["hero"] }));
    });
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: "rename voice" }));
    const input = screen.getByRole("textbox", { name: "voice name" });
    await user.clear(input);
    await user.type(input, "The Narrator");
    // the post-save invalidation refetches the library — carry the rename on that GET
    server.get("/api/voices", { voices: [makeVoice({ voice_id: "v1", name: "The Narrator", tags: ["hero"] })], unreadable: [] });
    await user.click(screen.getByRole("button", { name: "save" }));

    // the PATCH carries name ONLY — never recipe/seed/tags
    await waitFor(() => expect(server.jsonBodyOf("PATCH", "/api/voices/v1")).toEqual({ name: "The Narrator" }));
    expect(await screen.findByText("The Narrator")).toBeInTheDocument();
  });

  it("duplicate opens Add pre-filled with the blend recipe + seed and creates a NEW voice", async () => {
    const user = userEvent.setup();
    const server = mountServer({
      voices: [
        makeVoice({
          voice_id: "vb",
          name: "Duet",
          kind: "blend",
          preset_id: null,
          seed: 999,
          blend: [
            { preset_id: "af_heart", weight: 0.6 },
            { preset_id: "af_nicole", weight: 0.4 },
          ],
        }),
      ],
    })
      .get("/api/engines/kokoro/voices", kokoroCatalog)
      .post("/api/voices", makeVoice({ voice_id: "vb-copy", kind: "blend" }));
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: "duplicate voice" }));
    const dialog = await screen.findByRole("dialog", { name: "Duplicate voice" });
    // pre-filled: copy name, and it lands in manual-mix mode with the source's layers
    expect(within(dialog).getByDisplayValue("Duet copy")).toBeInTheDocument();
    expect(within(dialog).getByText(/the original stays exactly as it is/)).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "create copy" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    // a fresh voice from the same recipe, carrying the source seed so it starts identical
    expect(server.jsonBodyOf("POST", "/api/voices")).toEqual({
      kind: "blend",
      name: "Duet copy",
      components: [
        { preset_id: "af_heart", weight: 60 },
        { preset_id: "af_nicole", weight: 40 },
      ],
      seed: 999,
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
    // plain words and the price — the raw code is only a tooltip
    expect(await screen.findByText(/billed per use — this sample costs about \$0\.0123/)).toBeInTheDocument();
    expect(screen.queryByText("payment_confirmation_required")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /confirm ~\$0\.0123 & play/ }));

    await waitFor(() => expect(screen.queryByText(/billed per use/)).not.toBeInTheDocument());
    const posts = server.calls.filter((c) => c.method === "POST" && c.url.includes("/audition"));
    expect(posts).toHaveLength(2);
    expect(JSON.parse(posts[0].body as string)).toEqual({ confirm_paid: false });
    expect(JSON.parse(posts[1].body as string)).toEqual({ confirm_paid: true });
  });

  it("engine_cold refusal offers warm-up: POSTs the voice's engine warmup and resets to the audition control", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "k1", name: "Cold Heart" })] });
    server.error("POST", "/api/voices/k1/audition", 409, "engine_cold", "kokoro has no warm worker");
    server.post("/api/engines/kokoro/warmup", makeJob({ job_id: "w1", kind: "warmup", state: "running" }));
    server.get("/api/jobs/w1", makeJob({ job_id: "w1", kind: "warmup", state: "running" }));
    renderWithProviders(<Voices />);

    await user.click(await screen.findByRole("button", { name: /^audition/ }));
    expect(await screen.findByText(/Kokoro hasn't been downloaded yet/)).toBeInTheDocument();

    // the recourse targets the VOICE's engine, and success clears back to an idle audition
    await user.click(screen.getByRole("button", { name: "Download now" }));
    await waitFor(() => expect(server.lastCall("POST", "/warmup")?.url).toBe("/api/engines/kokoro/warmup"));
    expect(await screen.findByRole("button", { name: /^audition/ })).toBeInTheDocument();
    expect(screen.queryByText(/hasn't been downloaded yet/)).not.toBeInTheDocument();
    expect(screen.getByText(/Downloading Kokoro in the background/)).toBeInTheDocument();
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
      expect(screen.getByText(/billed per use/)).toBeInTheDocument();
      expect(bodies).toEqual([{ confirm_paid: false }]);

      // the user confirms the paid audition; the server then refuses softly (GPU borrowed)
      fireEvent.click(screen.getByRole("button", { name: /confirm ~\$0\.0123 & play/ }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });
      expect(bodies).toEqual([{ confirm_paid: false }, { confirm_paid: true }]);
      expect(screen.getByText(/Waiting for the recording to pause between lines/)).toBeInTheDocument();

      // bounded auto-retry with backoff: every re-POST must keep the user's confirm_paid=true
      for (let i = 0; i < 6; i++) {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(2000);
        });
      }
      expect(bodies).toHaveLength(5); // 1 unpaid + 1 explicit confirm + exactly 3 auto-retries
      expect(bodies.slice(1)).toEqual(Array(4).fill({ confirm_paid: true }));
      expect(screen.getByText(/hasn't paused for a sample yet/)).toBeInTheDocument();

      // ...and the budget is truly exhausted: no further auto-POST however long the clock runs
      await act(async () => {
        await vi.advanceTimersByTimeAsync(60_000);
      });
      expect(bodies).toHaveLength(5);

      // the manual retry also preserves the paid confirmation — never a fresh unpaid attempt
      fireEvent.click(screen.getByRole("button", { name: "try again" }));
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
    expect(screen.queryByText(/billed per use/)).not.toBeInTheDocument();
  });

  it("delete always asks first: the dialog lists each book by title and each part by character name", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Busy Voice" })] })
      .get("/api/books", {
        books: [
          makeBook({ book_id: "b1", title: "Pride and Prejudice" }),
          makeBook({ book_id: "b2", title: "Emma" }),
        ],
      })
      .get("/api/voices/v1/references", {
        voice_id: "v1",
        references: [
          { book_id: "b1", role: "narrator" },
          { book_id: "b1", role: "character:eliz" },
          { book_id: "b2", role: "thought" },
          { book_id: "b2", role: "character:ghost" },
        ],
      })
      .get("/api/books/b1/characters", charactersOf("b1", [{ id: "eliz", name: "Elizabeth Bennet" }]))
      .error("GET", "/api/books/b2/characters", 409, "not_attributed", "book b2 is not attributed");
    renderWithProviders(<Voices />);

    await screen.findByText("Busy Voice");
    await user.click(screen.getByRole("button", { name: "delete Busy Voice" }));

    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    const list = await within(dialog).findByRole("list", { name: "books using this voice" });
    await waitFor(() =>
      expect(within(list).getAllByRole("listitem")[0]).toHaveTextContent(
        "Pride and Prejudice — Narrator, Elizabeth Bennet",
      ),
    );
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    // b2's characters can't be read: a generic label, never the raw character id
    expect(items[1]).toHaveTextContent("Emma — Inner thoughts, A character");
    expect(dialog).not.toHaveTextContent("ghost");
    expect(dialog).not.toHaveTextContent("character:");
    expect(within(dialog).getByText(/still cast can't be deleted/)).toBeInTheDocument();
    // each book is the next step: a link straight to its Character Review
    expect(within(list).getByRole("link", { name: "Pride and Prejudice" })).toHaveAttribute(
      "href",
      "/review?book=b1",
    );
    // a cast voice can't be deleted: the confirm key stays off and nothing is sent
    expect(within(dialog).getByRole("button", { name: "delete voice" })).toBeDisabled();
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(0);
  });

  it("an unused preset voice says 'Not cast in any book'; confirm DELETEs exactly once and the card goes", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Solo Voice" })] })
      .get("/api/voices/v1/references", { voice_id: "v1", references: [] })
      .delete("/api/voices/v1", { deleted: "v1" });
    renderWithProviders(<Voices />);

    await screen.findByText("Solo Voice");
    await user.click(screen.getByRole("button", { name: "delete Solo Voice" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    expect(await within(dialog).findByText("Not cast in any book.")).toBeInTheDocument();
    expect(dialog).toHaveTextContent("This can't be undone.");
    expect(dialog).not.toHaveTextContent(/reference clip/);
    // "not cast" is all /references can prove: past recordings are spoken for in plain words
    expect(dialog).toHaveTextContent(/already recorded with this voice keep playing/);
    // opening the dialog sends nothing destructive
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(0);

    server.get("/api/voices", { voices: [], unreadable: [] }); // the post-delete refetch finds an empty booth
    const confirm = within(dialog).getByRole("button", { name: "delete voice" });
    expect(confirm).toHaveClass("danger");
    await user.click(confirm);

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await waitFor(() => expect(screen.queryByText("Solo Voice")).not.toBeInTheDocument());
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(1);
    expect(server.lastCall("DELETE", "/api/voices/v1")).toBeDefined();
    expect(await screen.findByText(/no voices yet/)).toBeInTheDocument();
  });

  it("a cloned voice's dialog warns the consent-attested reference clip is deleted permanently", async () => {
    const user = userEvent.setup();
    mountServer({
      voices: [
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
    }).get("/api/voices/vc/references", { voice_id: "vc", references: [] });
    renderWithProviders(<Voices />);

    await screen.findByText("Mr. Darcy");
    await user.click(screen.getByRole("button", { name: "delete Mr. Darcy" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    expect(dialog).toHaveTextContent(/reference clip this voice was cloned from/);
    expect(dialog).toHaveTextContent(/attested consent for — is deleted permanently/);
    expect(dialog).toHaveTextContent("This can't be undone.");
  });

  it("cancel closes the dialog and deletes nothing", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Keep Me" })] }).get(
      "/api/voices/v1/references",
      { voice_id: "v1", references: [] },
    );
    renderWithProviders(<Voices />);

    await screen.findByText("Keep Me");
    await user.click(screen.getByRole("button", { name: "delete Keep Me" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    await within(dialog).findByText("Not cast in any book.");
    await user.click(within(dialog).getByRole("button", { name: "cancel" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(0);
    expect(screen.getByText("Keep Me")).toBeInTheDocument();
  });

  it("a render_active refusal reads in plain words, keeps the dialog open, and is not retried", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Busy Booth" })] })
      .get("/api/voices/v1/references", { voice_id: "v1", references: [] })
      .error("DELETE", "/api/voices/v1", 409, "render_active", "a render job is running; wait or cancel it first");
    renderWithProviders(<Voices />);

    await screen.findByText("Busy Booth");
    await user.click(screen.getByRole("button", { name: "delete Busy Booth" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    await within(dialog).findByText("Not cast in any book.");
    await user.click(within(dialog).getByRole("button", { name: "delete voice" }));

    expect(
      await within(dialog).findByText("A recording or audiobook build is running, so nothing was deleted."),
    ).toBeInTheDocument();
    expect(within(dialog).queryByText("render_active")).not.toBeInTheDocument();
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(1);
    expect(screen.getByRole("dialog", { name: "Delete voice" })).toBeInTheDocument();
  });

  it("a voice cast while the dialog was open: the pre-delete check catches it, refreshes the list, sends no DELETE", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Racy" })] })
      .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Persuasion" })] })
      .get("/api/voices/v1/references", { voice_id: "v1", references: [] });
    renderWithProviders(<Voices />);

    await screen.findByText("Racy");
    await user.click(screen.getByRole("button", { name: "delete Racy" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    await within(dialog).findByText("Not cast in any book.");
    // someone casts it as narrator before the click lands
    server.get("/api/voices/v1/references", { voice_id: "v1", references: [{ book_id: "b1", role: "narrator" }] });
    await user.click(within(dialog).getByRole("button", { name: "delete voice" }));

    expect(await within(dialog).findByText("This voice is cast in a book, so it wasn't deleted.")).toBeInTheDocument();
    expect(await within(dialog).findByText("Persuasion")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "delete voice" })).toBeDisabled();
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(0);
  });

  it("an unreadable casting file fails closed: plain explanation, delete stays off", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Unsure" })] }).error(
      "GET",
      "/api/voices/v1/references",
      500,
      "corrupt_artifact",
      "cannot verify voice references: output/b1/assignments.json is unreadable",
    );
    renderWithProviders(<Voices />);

    await screen.findByText("Unsure");
    await user.click(screen.getByRole("button", { name: "delete Unsure" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    expect(
      await within(dialog).findByText(
        "Seiyuu couldn't check which books use this voice, so it can't be deleted right now.",
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "delete voice" })).toBeDisabled();
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(0);
  });

  it("a voice cast between the check and the DELETE: the server's 409 reads in plain words and refreshes the list", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Racy" })] })
      .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Persuasion" })] })
      .get("/api/voices/v1/references", { voice_id: "v1", references: [] });
    // the dialog's list AND the hook's pre-delete check both see it unused; the cast lands
    // just before the DELETE, so only the server catches it
    server.on("DELETE", "/api/voices/v1", () => {
      server.get("/api/voices/v1/references", { voice_id: "v1", references: [{ book_id: "b1", role: "narrator" }] });
      return errorResponse(409, "voice_referenced", "voice 'v1' is still assigned in: b1 (narrator)");
    });
    renderWithProviders(<Voices />);

    await screen.findByText("Racy");
    await user.click(screen.getByRole("button", { name: "delete Racy" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    await within(dialog).findByText("Not cast in any book.");
    await user.click(within(dialog).getByRole("button", { name: "delete voice" }));

    expect(await within(dialog).findByText("This voice is cast in a book, so it wasn't deleted.")).toBeInTheDocument();
    expect(await within(dialog).findByRole("link", { name: "Persuasion" })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "delete voice" })).toBeDisabled();
    expect(within(dialog).queryByText("voice_referenced")).not.toBeInTheDocument();
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(1);
    // the card is still there (hidden from the a11y tree only while the modal is up)
    expect(screen.getByRole("button", { name: "delete Racy", hidden: true })).toBeInTheDocument();
  });

  it("a DELETE that finds the voice already gone drops the stale card", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Ghost" })] })
      .get("/api/voices/v1/references", { voice_id: "v1", references: [] })
      .error("DELETE", "/api/voices/v1", 404, "not_found", "voice 'v1' not found");
    renderWithProviders(<Voices />);

    await screen.findByText("Ghost");
    await user.click(screen.getByRole("button", { name: "delete Ghost" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    await within(dialog).findByText("Not cast in any book.");
    server.get("/api/voices", { voices: [], unreadable: [] }); // the library refetch no longer has it
    await user.click(within(dialog).getByRole("button", { name: "delete voice" }));

    await waitFor(() => expect(screen.queryByText("Ghost")).not.toBeInTheDocument());
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(1);
  });

  it("opening the dialog on a voice deleted elsewhere drops the stale card", async () => {
    const user = userEvent.setup();
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Gone Elsewhere" })] }).error(
      "GET",
      "/api/voices/v1/references",
      404,
      "not_found",
      "voice 'v1' not found",
    );
    renderWithProviders(<Voices />);

    await screen.findByText("Gone Elsewhere");
    server.get("/api/voices", { voices: [], unreadable: [] });
    await user.click(screen.getByRole("button", { name: "delete Gone Elsewhere" }));

    await waitFor(() => expect(screen.queryByText("Gone Elsewhere")).not.toBeInTheDocument());
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(0);
  });

  it("while the DELETE is in flight the dialog can't be closed, so its result is always shown", async () => {
    const user = userEvent.setup();
    let release: (r: Response) => void = () => {};
    const server = mountServer({ voices: [makeVoice({ voice_id: "v1", name: "Slow Booth" })] }).get(
      "/api/voices/v1/references",
      { voice_id: "v1", references: [] },
    );
    server.on(
      "DELETE",
      "/api/voices/v1",
      () =>
        new Promise<Response>((resolve) => {
          release = resolve;
        }),
    );
    renderWithProviders(<Voices />);

    await screen.findByText("Slow Booth");
    await user.click(screen.getByRole("button", { name: "delete Slow Booth" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete voice" });
    await within(dialog).findByText("Not cast in any book.");
    await user.click(within(dialog).getByRole("button", { name: "delete voice" }));

    expect(await within(dialog).findByRole("button", { name: "deleting…" })).toBeDisabled();
    expect(within(dialog).getByRole("button", { name: "cancel" })).toBeDisabled();
    expect(within(dialog).queryByRole("button", { name: "esc" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.getByRole("dialog", { name: "Delete voice" })).toBeInTheDocument();

    await act(async () => {
      release(errorResponse(409, "render_active", "a render job is running; wait or cancel it first"));
    });
    expect(
      await within(dialog).findByText("A recording or audiobook build is running, so nothing was deleted."),
    ).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "cancel" })).toBeEnabled();
  });

  it("each card's delete key names its own voice", async () => {
    mountServer({
      voices: [makeVoice({ voice_id: "v1", name: "Alpha" }), makeVoice({ voice_id: "v2", name: "Beta" })],
    });
    renderWithProviders(<Voices />);

    expect(await screen.findByRole("button", { name: "delete Alpha" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "delete Beta" })).toBeInTheDocument();
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
