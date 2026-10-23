import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type {
  BookCard,
  BookDetail,
  CostEstimateOut,
  QuoteResponse,
  RenderSummaryOut,
  SystemStatusOut,
  ValidationRow,
  VoiceOut,
} from "../api/types";
import { makeBook, makeJob } from "../test/fixtures";
import { errorResponse, jsonResponse, mockApi, renderWithProviders, type MockApi } from "../test/utils";
import { RenderJobs } from "./RenderJobs";

/* ---- screen-specific fixtures (shapes from api/types) -------------------------------- */

function makeDetail(card: BookCard, chapterCount = 5): BookDetail {
  return {
    status: card,
    chapters: Array.from({ length: chapterCount }, (_, i) => ({
      index: i + 1,
      title: `Chapter ${i + 1}`,
      blocks: 40,
      speakable_blocks: 30,
    })),
    runtime_estimate_seconds: 7200,
    active_job: card.active_job,
    recent_jobs: [],
    downloads: { m4b: null, chapter_mp3s: [] },
    cover: null,
  };
}

function makeSystem(over: Partial<SystemStatusOut> = {}): SystemStatusOut {
  return {
    attribution: {
      provider: "local",
      model: "qwen2.5:7b",
      anthropic_model: "claude-sonnet-4",
      prompt_version: "v6",
      hybrid: false,
    },
    keys: { anthropic_configured: true, elevenlabs_configured: true },
    apply_emotion: false,
    ...over,
  };
}

function makeEstimate(over: Partial<CostEstimateOut> = {}): CostEstimateOut {
  return {
    total_usd: 0,
    paid_segments: 0,
    cached_segments: 0,
    free_segments: 150,
    fingerprint: "fp-1",
    assignment_hash: "ah-1",
    mode: "multivoice",
    chapters: [],
    edit_warnings: [],
    ...over,
  };
}

function makeSummary(over: Partial<RenderSummaryOut> = {}): RenderSummaryOut {
  return {
    book_id: "demo",
    mode: "multivoice",
    chapters: [{ index: 1, title: "Chapter 1", segments: 300, duration_seconds: 1800 }],
    total_seconds: 1800,
    voices_used: {},
    validation_failures: 0,
    rendered_assignment_hash: null,
    active_mode: "multi",
    available_modes: ["multi"],
    ...over,
  };
}

function makeQuote(over: Partial<QuoteResponse> = {}): QuoteResponse {
  const now = Math.floor(Date.now() / 1000);
  return {
    token: "tok-1",
    book_id: "demo",
    chapters: [],
    total_usd: 1.25,
    paid_segments: 7,
    fingerprint: "fp-1",
    assignment_hash: "ah-1",
    issued_at: now,
    expires_at: now + 900,
    ttl_seconds: 900,
    max_usd_ceiling: 2.5,
    ...over,
  };
}

/** Mount routes for one book; per-test routes (estimate/quotes/render) layer on top. */
function mountRoutes(card: BookCard, system: SystemStatusOut = makeSystem()): MockApi {
  return mockApi()
    .get("/api/books", { books: [card] })
    .get("/api/books/demo", makeDetail(card))
    .get("/api/system", system)
    // One narrator's voice/speed picker (fetched only in that mode, or to name checked voices)
    .get("/api/voices", { voices: [makeVoice()], unreadable: [] })
    .get("/api/engines/kokoro/voices", {
      engine_id: "kokoro",
      voices: [
        { id: "af_heart", name: "Heart", language: "en-us", gender: "female", description: "warm" },
        { id: "am_adam", name: "Adam", language: "en-us", gender: "male", description: null },
      ],
    })
    .get("/api/settings", { kokoro_default_voice: "af_heart", attribution_provider: "local", attribution_model: "qwen2.5:7b" })
    // a failed audio check skips words the lexicon already respells
    .get("/api/books/demo/lexicon", { book_id: "demo", schema_version: 1, entries: [], suggestions: [] });
}

function makeVoice(over: Partial<VoiceOut> = {}): VoiceOut {
  return {
    voice_id: "v-kira",
    name: "Kira",
    kind: "cloned",
    engine: "chatterbox",
    preset_id: null,
    blend: null,
    reference_audio: "reference.wav",
    seed: 41172,
    consent_attested: true,
    consent: null,
    tags: [],
    created_at: "2026-07-01",
    has_audition: true,
    ...over,
  };
}

/** ingested + attributed + assigned -> the screen defaults to a ready multivoice render. */
const readyCard = () => makeBook({ attributed: true, assigned: true });

const postCount = (server: MockApi, path: string) =>
  server.calls.filter((c) => c.method === "POST" && c.url.includes(path)).length;

/* ---- tests ---------------------------------------------------------------------------- */

describe("RenderJobs", () => {
  it("fetches the estimate for the default multivoice mode (emotion default threaded) and shows the price", async () => {
    const server = mountRoutes(readyCard()).get(
      "/api/books/demo/cost-estimate",
      makeEstimate({ total_usd: 1.25, paid_segments: 7, cached_segments: 90, free_segments: 43 }),
    );
    renderWithProviders(<RenderJobs />);

    expect(
      await screen.findByText(/7 line\(s\) use a paid cloud voice — about \$1\.25\./),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /get the price — about \$1\.25/ })).toBeInTheDocument();
    const est = server.lastCall("GET", "/cost-estimate");
    expect(est?.url).toContain("mode=multivoice");
    expect(est?.url).toContain("apply_emotion=false");
  });

  it("a server default of apply_emotion=true flows into the estimate and quote with no checkbox touch", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard(), makeSystem({ apply_emotion: true }))
      .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7 }))
      .post("/api/books/demo/quotes", makeQuote());
    renderWithProviders(<RenderJobs />);

    // the hint proves the default came from /api/system rather than a hardcoded false
    expect(await screen.findByText(/default: on/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "voice per-line emotion" })).toBeChecked();
    await waitFor(() =>
      expect(server.lastCall("GET", "/cost-estimate")?.url).toContain("apply_emotion=true"),
    );

    // mint WITHOUT touching the toggle: the server default must reach the quote fingerprint
    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    await screen.findByRole("button", { name: "CONFIRM & RENDER" });
    expect(server.jsonBodyOf("POST", "/quotes")).toEqual({
      mode: "multivoice",
      chapters: [],
      apply_emotion: true,
    });
  });

  it("single-voice mode is emotion-agnostic: no apply_emotion anywhere, and the render carries single:{}", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard())
      .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7, mode: "single" }))
      .post("/api/books/demo/quotes", makeQuote({ token: "tok-s" }))
      .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: "One narrator" }));
    // the emotion toggle is a multivoice affordance — single hides it entirely
    await waitFor(() =>
      expect(screen.queryByRole("checkbox", { name: "voice per-line emotion" })).not.toBeInTheDocument(),
    );
    // ...and the estimate URL carries NO apply_emotion param (cache-key parity with the backend)
    await waitFor(() => {
      const est = server.lastCall("GET", "/cost-estimate");
      expect(est?.url).toContain("mode=single");
      expect(est?.url).not.toContain("apply_emotion");
    });

    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    const confirm = await screen.findByRole("button", { name: "CONFIRM & RENDER" });
    expect(server.jsonBodyOf("POST", "/quotes")).toEqual({ mode: "single", chapters: [], single: {} });

    await user.click(confirm);
    await waitFor(() => expect(server.lastCall("POST", "/render")).toBeDefined());
    expect(server.jsonBodyOf("POST", "/render")).toEqual({
      mode: "single",
      chapters: [],
      single: {},
      cost_token: "tok-s",
    });
  });

  it("a free estimate renders in one click: POST /render directly, no quote minted", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard())
      .get("/api/books/demo/cost-estimate", makeEstimate())
      .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
    renderWithProviders(<RenderJobs />);

    await user.click(
      await screen.findByRole("button", { name: "record — free, nothing to approve" }),
    );

    await waitFor(() => expect(server.lastCall("POST", "/render")).toBeDefined());
    expect(server.jsonBodyOf("POST", "/render")).toEqual({
      mode: "multivoice",
      chapters: [],
      apply_emotion: false,
    });
    expect(server.lastCall("POST", "/quotes")).toBeUndefined();
  });

  it("a paid estimate never auto-runs: mint POSTs /quotes, only CONFIRM & RENDER POSTs /render with the token", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard())
      .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7 }))
      .post("/api/books/demo/quotes", makeQuote({ token: "tok-abc123" }))
      .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));

    // the ticket is live but nothing rendered yet — paid work waits for the explicit confirm
    const confirm = await screen.findByRole("button", { name: "CONFIRM & RENDER" });
    expect(server.jsonBodyOf("POST", "/quotes")).toEqual({
      mode: "multivoice",
      chapters: [],
      apply_emotion: false,
    });
    expect(server.lastCall("POST", "/render")).toBeUndefined();

    await user.click(confirm);
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "CONFIRM & RENDER" })).not.toBeInTheDocument(),
    );
    expect(server.jsonBodyOf("POST", "/render")).toEqual({
      mode: "multivoice",
      chapters: [],
      apply_emotion: false,
      cost_token: "tok-abc123",
    });
  });

  it("the emotion toggle threads apply_emotion=true into the estimate URL, the quote body, AND the render body (parity)", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard())
      .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7 }))
      .post("/api/books/demo/quotes", makeQuote())
      .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("checkbox", { name: "voice per-line emotion" }));
    await waitFor(() =>
      expect(server.lastCall("GET", "/cost-estimate")?.url).toContain("apply_emotion=true"),
    );

    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    const confirm = await screen.findByRole("button", { name: "CONFIRM & RENDER" });
    expect((server.jsonBodyOf("POST", "/quotes") as Record<string, unknown>).apply_emotion).toBe(true);

    await user.click(confirm);
    await waitFor(() => expect(server.lastCall("POST", "/render")).toBeDefined());
    expect((server.jsonBodyOf("POST", "/render") as Record<string, unknown>).apply_emotion).toBe(true);
  });

  it("a chapter-range scope lands as chapters params in the estimate query and in the minted quote body", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard());
    server.on("GET", /\/api\/books\/demo\/cost-estimate/, (url) =>
      jsonResponse(
        makeEstimate({
          total_usd: 1.25,
          paid_segments: 7,
          chapters: url.searchParams.getAll("chapters").map(Number),
        }),
      ),
    );
    server.post("/api/books/demo/quotes", makeQuote({ chapters: [1, 2] }));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: "chapter range" }));
    // controlled number input: clear() snaps back to the controlled value before type()
    // appends, so drive it with a single change event instead
    fireEvent.change(screen.getByRole("spinbutton", { name: "to" }), { target: { value: "2" } });

    await waitFor(() => {
      const est = server.lastCall("GET", "/cost-estimate");
      expect(est?.url).toContain("chapters=1&chapters=2");
      expect(est?.url).not.toContain("chapters=3");
    });

    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    await screen.findByRole("button", { name: "CONFIRM & RENDER" });
    expect(server.jsonBodyOf("POST", "/quotes")).toEqual({
      mode: "multivoice",
      chapters: [1, 2],
      apply_emotion: false,
    });
  });

  it("quote_expired on render re-mints silently: a fresh live ticket, and never an auto-render", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard()).get(
      "/api/books/demo/cost-estimate",
      makeEstimate({ total_usd: 1.25, paid_segments: 7 }),
    );
    let minted = 0;
    server.on("POST", "/api/books/demo/quotes", () => jsonResponse(makeQuote({ token: `tok-${++minted}` })));
    server.on("POST", "/api/books/demo/render", () => errorResponse(409, "quote_expired", "the ticket lapsed"));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    await user.click(await screen.findByRole("button", { name: "CONFIRM & RENDER" }));

    await waitFor(() => expect(postCount(server, "/quotes")).toBe(2));
    // the fresh ticket is live again (not stamped) and still waits for an explicit confirm
    expect(await screen.findByRole("button", { name: "CONFIRM & RENDER" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "RE-MINT QUOTE" })).not.toBeInTheDocument();
    expect(postCount(server, "/render")).toBe(1);
  });

  it("a refused render start reads in plain words and offers to stop the job in the way, never retrying", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard())
      .get("/api/books/demo/cost-estimate", makeEstimate())
      .error(
        "POST",
        "/api/books/demo/render",
        409,
        "conflicting_job",
        "an attribute job for 'demo' is running; wait for it or cancel it before rendering",
        makeJob({ job_id: "job-a1", kind: "attribute", state: "running" }),
      )
      .post("/api/jobs/job-a1/cancel", makeJob({ job_id: "job-a1", kind: "attribute", state: "canceled" }));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: "record — free, nothing to approve" }));
    const line = await screen.findByText(/Finding the speakers is still running for this book/);
    expect(screen.queryByText(/an attribute job for 'demo'/)).not.toBeInTheDocument();
    await user.click(within(line).getByRole("button", { name: "Stop finding the speakers" }));
    await waitFor(() => expect(postCount(server, "/api/jobs/job-a1/cancel")).toBe(1));
    expect(postCount(server, "/render")).toBe(1);
  });

  it("cost_drift stamps the ticket dead (DRIFT + refusal reason) and RE-MINT revives a live ticket", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard()).get(
      "/api/books/demo/cost-estimate",
      makeEstimate({ total_usd: 1.25, paid_segments: 7 }),
    );
    server.post("/api/books/demo/quotes", makeQuote());
    server.on("POST", "/api/books/demo/render", () =>
      errorResponse(409, "cost_drift", "the cache shifted under this quote"),
    );
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    await user.click(await screen.findByRole("button", { name: "CONFIRM & RENDER" }));

    expect(await screen.findByText("DRIFT")).toBeInTheDocument();
    // why, in plain words; the server's own text stays in the tooltip
    const why = screen.getByText(/The approved price no longer matches/);
    expect(why).toHaveAttribute("title", "the cache shifted under this quote");
    expect(screen.queryByRole("button", { name: "CONFIRM & RENDER" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "RE-MINT QUOTE" }));
    expect(await screen.findByRole("button", { name: "CONFIRM & RENDER" })).toBeInTheDocument();
    expect(postCount(server, "/quotes")).toBe(2);
  });

  it("a whole-book refusal opens the confirm dialog and confirming re-sends confirm_full=true", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard()).get("/api/books/demo/cost-estimate", makeEstimate());
    let renders = 0;
    server.on("POST", "/api/books/demo/render", () => {
      renders += 1;
      return renders === 1
        ? errorResponse(409, "full_render_confirmation_required", "confirm the full render", {
            speakable_blocks: 4200,
            runtime_estimate_seconds: 25200,
          })
        : jsonResponse(makeJob({ kind: "render", state: "queued" }));
    });
    renderWithProviders(<RenderJobs />);

    await user.click(
      await screen.findByRole("button", { name: "record — free, nothing to approve" }),
    );

    const dialog = await screen.findByRole("dialog", { name: "Record the whole book" });
    expect(dialog).toHaveTextContent(/4,?200/); // the refusal detail (segment count) is shown
    await user.click(screen.getByRole("button", { name: "record the whole book" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.jsonBodyOf("POST", "/render")).toEqual({
      mode: "multivoice",
      chapters: [],
      apply_emotion: false,
      confirm_full: true,
    });
  });

  it("a paid whole-book refusal re-sends BOTH the live ticket's token AND confirm_full on the confirmed retry", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard())
      .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7 }))
      .post("/api/books/demo/quotes", makeQuote({ token: "tok-x" }));
    let renders = 0;
    server.on("POST", "/api/books/demo/render", () => {
      renders += 1;
      return renders === 1
        ? errorResponse(409, "full_render_confirmation_required", "confirm the full render", {
            speakable_blocks: 4200,
            runtime_estimate_seconds: 25200,
          })
        : jsonResponse(makeJob({ kind: "render", state: "queued" }));
    });
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    await user.click(await screen.findByRole("button", { name: "CONFIRM & RENDER" }));

    // the whole-book gate fires ON TOP of the paid flow — the live ticket moves into the one
    // sheet (no second dialog) and ITS confirm is the only key
    const dialog = await screen.findByRole("dialog", { name: "Record the whole book" });
    expect(within(dialog).queryByRole("button", { name: "record the whole book" })).not.toBeInTheDocument();
    expect(within(dialog).getByText("$1.25")).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: "CONFIRM & RENDER" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(postCount(server, "/render")).toBe(2);
    expect(server.jsonBodyOf("POST", "/render")).toEqual({
      mode: "multivoice",
      chapters: [],
      apply_emotion: false,
      cost_token: "tok-x",
      confirm_full: true,
    });
    // success consumes the ticket: no live quote lingers to be double-confirmed
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "CONFIRM & RENDER" })).not.toBeInTheDocument(),
    );
  });

  it("paid attribution 402s without confirm_paid; the explicit confirm re-sends confirm_paid=true", async () => {
    const user = userEvent.setup();
    const card = makeBook(); // ingested, not attributed, not assigned -> single mode + attribution refusal
    const server = mockApi()
      .get("/api/books", { books: [card] })
      .get("/api/books/demo", makeDetail(card))
      .get(
        "/api/system",
        makeSystem({
          attribution: {
            provider: "anthropic",
            model: "claude-sonnet-4",
            anthropic_model: "claude-sonnet-4",
            prompt_version: "v6",
            hybrid: false,
          },
        }),
      )
      .get("/api/books/demo/cost-estimate", makeEstimate({ mode: "single" }));
    const bodies: Record<string, unknown>[] = [];
    server.on("POST", "/api/books/demo/attribute", (_url, init) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      bodies.push(body);
      return body.confirm_paid
        ? jsonResponse(makeJob({ kind: "attribute", state: "queued" }))
        : errorResponse(402, "payment_confirmation_required", "anthropic attribution is a paid run");
    });
    renderWithProviders(<RenderJobs />);

    // wait for the system defaults so the enqueue carries provider + model
    await screen.findByText(/Claude online · claude-sonnet-4/);
    await user.click(screen.getByRole("button", { name: /find the speakers in the whole book/ }));

    // the 402 surfaces the reason and an explicit confirm control — nothing auto-retries
    expect(await screen.findByText(/uses Claude online, which costs money/)).toBeInTheDocument();
    expect(screen.queryByText(/anthropic attribution is a paid run/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "confirm the paid run" }));

    await waitFor(() => expect(bodies).toHaveLength(2));
    expect(bodies[0]).toEqual({ chapters: [], provider: "anthropic", model: "claude-sonnet-4" });
    expect(bodies[1]).toEqual({
      chapters: [],
      provider: "anthropic",
      model: "claude-sonnet-4",
      confirm_paid: true,
    });
  });

  it("an Ollama pre-flight refusal reads in plain words with the fix; the raw message stays under Details", async () => {
    const user = userEvent.setup();
    const card = makeBook();
    mockApi()
      .get("/api/books", { books: [card] })
      .get("/api/books/demo", makeDetail(card))
      .get("/api/system", makeSystem())
      .get("/api/books/demo/cost-estimate", makeEstimate({ mode: "single" }))
      .error(
        "POST",
        "/api/books/demo/attribute",
        503,
        "ollama_unreachable",
        "Ollama isn't reachable at http://localhost:11434 - start it with 'ollama serve' ([WinError 10061])",
      );
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: /find the speakers in the whole book/ }));
    expect(await screen.findByText(/couldn't reach Ollama/)).toBeInTheDocument();
    expect(screen.getByText("ollama serve")).toBeInTheDocument();
    expect(screen.getByText("Details")).toBeInTheDocument();
    expect(screen.getByText(/WinError 10061/).closest("details")).not.toBeNull(); // never the primary text
  });

  it("once speakers are found the step is a done card with the re-run under Advanced, not a start button", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard()).get("/api/books/demo/cost-estimate", makeEstimate());
    renderWithProviders(<RenderJobs />, { route: "/render?book=demo" });

    const box = await screen.findByRole("region", { name: "Speakers found" });
    expect(box).toHaveAttribute("id", "find-speakers");
    expect(within(box).getByRole("link", { name: "check them on Speakers & cast" })).toHaveAttribute(
      "href",
      "/review?book=demo",
    );
    expect(within(box).queryByRole("button", { name: /find the speakers in/ })).not.toBeInTheDocument();
    // the provenance is only fetched once the Advanced disclosure is opened
    expect(server.lastCall("GET", "/characters")).toBeUndefined();
    server.get("/api/books/demo/characters", {
      book_id: "demo",
      provider_id: "local",
      model_id: "qwen2.5:7b",
      prompt_version: "v6",
      narration_segments: 1,
      low_confidence_segments: 0,
      unattributed_quote_segments: 0,
      confidence_threshold: 0.7,
      characters: [],
      flagged: [],
      notes: [],
      edit_warnings: [],
    });
    await user.click(within(box).getByText("Advanced: re-run speaker finding for chapters…"));
    expect(await within(box).findByRole("textbox", { name: "chapters to re-read" })).toBeInTheDocument();
    await waitFor(() => expect(server.lastCall("GET", "/api/books/demo/characters")).toBeDefined());
  });

  /* ---- active render mode (instant fallback) ------------------------------------------ */

  const modeGroup = async () => within(await screen.findByRole("group", { name: "version used for Listen" }));
  const renderedCard = () => makeBook({ attributed: true, assigned: true, rendered: true });
  /** Routes every rendered book mounts (a rendered card also polls the validation report). */
  const mountRendered = (card: BookCard) =>
    mountRoutes(card)
      .get("/api/books/demo/cost-estimate", makeEstimate())
      .get("/api/books/demo/validation", { validated_segments: 0, validation_failures: 0, results: [] });

  it("switching the active mode POSTs /render/mode and the refetched summary flips the active chip", async () => {
    const user = userEvent.setup();
    const server = mountRendered(renderedCard())
      .get("/api/books/demo/render", makeSummary({ available_modes: ["single", "multi"] }))
      .post("/api/books/demo/render/mode", makeSummary({ mode: "single", active_mode: "single", available_modes: ["single", "multi"] }));
    renderWithProviders(<RenderJobs />);

    const group = await modeGroup();
    expect(group.getByRole("button", { name: "Full cast · in use" })).toBeDisabled();
    const switchBtn = group.getByRole("button", { name: "One narrator · use this" });
    expect(switchBtn).toBeEnabled();

    // the flipped chip may only appear via the post-switch invalidation refetching the
    // summary — re-register the GET so the refetch (and nothing else) carries it
    server.get(
      "/api/books/demo/render",
      makeSummary({ mode: "single", active_mode: "single", available_modes: ["single", "multi"] }),
    );
    await user.click(switchBtn);

    expect(await group.findByRole("button", { name: "One narrator · in use" })).toBeDisabled();
    expect(group.getByRole("button", { name: "Full cast · use this" })).toBeEnabled();
    expect(server.jsonBodyOf("POST", "/render/mode")).toEqual({ mode: "single" });
  });

  it("the mode control never mounts before the first render", async () => {
    const server = mountRoutes(readyCard()).get("/api/books/demo/cost-estimate", makeEstimate());
    renderWithProviders(<RenderJobs />);

    await screen.findByRole("button", { name: "record — free, nothing to approve" });
    expect(screen.queryByRole("group", { name: "version used for Listen" })).not.toBeInTheDocument();
    expect(server.lastCall("GET", "/api/books/demo/render")).toBeUndefined();
  });

  it("a 409 conflicting_job on the switch renders the refusal with the envelope message", async () => {
    const user = userEvent.setup();
    const server = mountRendered(renderedCard()).get(
      "/api/books/demo/render",
      makeSummary({ available_modes: ["single", "multi"] }),
    );
    server.error(
      "POST",
      "/api/books/demo/render/mode",
      409,
      "conflicting_job",
      "a render job for 'demo' is running; wait for it or cancel it before switching the active render mode",
    );
    renderWithProviders(<RenderJobs />);

    await user.click((await modeGroup()).getByRole("button", { name: "One narrator · use this" }));

    expect(await screen.findByText("busy")).toBeInTheDocument();
    expect(
      screen.getByText(/a render job for 'demo' is running; wait for it or cancel it/),
    ).toBeInTheDocument();
  });

  it("with only one mode rendered the other chip reads not-rendered and is disabled", async () => {
    mountRendered(renderedCard()).get(
      "/api/books/demo/render",
      makeSummary({ mode: "single", active_mode: "single", available_modes: ["single"] }),
    );
    renderWithProviders(<RenderJobs />);

    const group = await modeGroup();
    expect(group.getByRole("button", { name: "One narrator · in use" })).toBeDisabled();
    expect(group.getByRole("button", { name: "Full cast · not recorded" })).toBeDisabled();
    expect(screen.getByText(/record the other version once and you can switch/)).toBeInTheDocument();
  });

  it("while a render job runs both chips deaden and the hint says the job owns the render", async () => {
    const card = {
      ...renderedCard(),
      active_job: { job_id: "j-1", kind: "render" as const, state: "running" as const },
    };
    mountRendered(card).get("/api/books/demo/render", makeSummary({ available_modes: ["single", "multi"] }));
    renderWithProviders(<RenderJobs />);

    const group = await modeGroup();
    expect(group.getByRole("button", { name: "One narrator · use this" })).toBeDisabled();
    expect(
      screen.getByText(/Recording is running — wait for it or stop it first/),
    ).toBeInTheDocument();
  });

  /* ---- re-render (force) + stale-cast banner ------------------------------------------ */

  const forceToggle = () =>
    screen.findByRole("checkbox", { name: "record again from scratch — ignore saved audio" });

  it("the force toggle threads force=true into the estimate URL and the render body", async () => {
    const user = userEvent.setup();
    const server = mountRendered(renderedCard())
      .get("/api/books/demo/render", makeSummary())
      .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
    renderWithProviders(<RenderJobs />);

    await user.click(await forceToggle());
    await waitFor(() =>
      expect(server.lastCall("GET", "/cost-estimate")?.url).toContain("force=true"),
    );

    await user.click(
      await screen.findByRole("button", { name: "record — free, nothing to approve" }),
    );
    await waitFor(() => expect(server.lastCall("POST", "/render")).toBeDefined());
    expect(server.jsonBodyOf("POST", "/render")).toEqual({
      mode: "multivoice",
      chapters: [],
      apply_emotion: false,
      force: true,
    });
  });

  it("the force toggle only appears once a render exists (nothing to re-render otherwise)", async () => {
    mountRoutes(readyCard()).get("/api/books/demo/cost-estimate", makeEstimate());
    renderWithProviders(<RenderJobs />);
    await screen.findByRole("button", { name: "record — free, nothing to approve" });
    expect(
      screen.queryByRole("checkbox", { name: "record again from scratch — ignore saved audio" }),
    ).not.toBeInTheDocument();
  });

  it("warns that the cast changed when the rendered assignment differs from the current one", async () => {
    // estimate default assignment_hash is "ah-1"; the render was built with "ah-OLD" -> stale
    mountRendered(renderedCard()).get(
      "/api/books/demo/render",
      makeSummary({ rendered_assignment_hash: "ah-OLD" }),
    );
    renderWithProviders(<RenderJobs />);
    expect(
      await screen.findByText(/the current cast differs from the recorded audio/),
    ).toBeInTheDocument();
  });

  it("no cast-changed warning when the rendered assignment matches the current one", async () => {
    mountRendered(renderedCard()).get(
      "/api/books/demo/render",
      makeSummary({ rendered_assignment_hash: "ah-1" }),
    );
    renderWithProviders(<RenderJobs />);
    await forceToggle(); // the rendered summary has loaded
    expect(
      screen.queryByText(/the current cast differs from the recorded audio/),
    ).not.toBeInTheDocument();
  });

  /* ---- one-click finish (assemble → master) ------------------------------------------- */

  it("Build the .m4b builds only the .m4b by default — master, no chapter MP3s — and reports done", async () => {
    const user = userEvent.setup();
    const server = mountRendered(renderedCard())
      .get("/api/books/demo/render", makeSummary())
      .post("/api/books/demo/master", makeJob({ job_id: "j-mst", kind: "master", state: "queued" }));
    const { queryClient } = renderWithProviders(<RenderJobs />);

    const build = await screen.findByRole("button", { name: "Build the .m4b" });
    expect(screen.getByRole("checkbox", { name: "also make chapter MP3s (optional)" })).not.toBeChecked();
    await waitFor(() => expect(build).toBeEnabled());
    await user.click(build);

    await waitFor(() => expect(postCount(server, "/master")).toBe(1));
    expect(postCount(server, "/assemble")).toBe(0);

    server.get("/api/jobs", {
      jobs: [makeJob({ job_id: "j-mst", kind: "master", state: "succeeded", is_terminal: true, finished_at: "2026-07-11T00:02:00Z" })],
    });
    void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    expect(await screen.findByText(/finished — the \.m4b is ready below/)).toBeInTheDocument();
    expect(postCount(server, "/assemble")).toBe(0);
  });

  it("with chapter MP3s ticked it runs assemble, waits for it to land, then master, and reports done", async () => {
    const user = userEvent.setup();
    const server = mountRendered(renderedCard())
      .get("/api/books/demo/render", makeSummary())
      .post("/api/books/demo/assemble", makeJob({ job_id: "j-asm", kind: "assemble", state: "queued" }));
    const { queryClient } = renderWithProviders(<RenderJobs />);

    // the key deadens until the book detail (status.rendered) lands
    const finishBtn = await screen.findByRole("button", { name: "Build the .m4b" });
    await waitFor(() => expect(finishBtn).toBeEnabled());
    await user.click(screen.getByRole("checkbox", { name: "also make chapter MP3s (optional)" }));
    await user.click(finishBtn);

    // assemble fires immediately; master must NOT — it waits for assemble to succeed
    await waitFor(() => expect(postCount(server, "/assemble")).toBe(1));
    expect(postCount(server, "/master")).toBe(0);
    expect(screen.getByText(/then the \.m4b — keep this/)).toBeInTheDocument();

    // the assemble job lands; the next poll advances the chain to master
    server.get("/api/jobs", {
      jobs: [makeJob({ job_id: "j-asm", kind: "assemble", state: "succeeded", is_terminal: true, finished_at: "2026-07-11T00:01:00Z" })],
    });
    server.post("/api/books/demo/master", makeJob({ job_id: "j-mst", kind: "master", state: "queued" }));
    void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    await waitFor(() => expect(postCount(server, "/master")).toBe(1));

    // the master job lands; the chain reports the m4b done
    server.get("/api/jobs", {
      jobs: [makeJob({ job_id: "j-mst", kind: "master", state: "succeeded", is_terminal: true, finished_at: "2026-07-11T00:02:00Z" })],
    });
    void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    expect(await screen.findByText(/finished — the \.m4b is ready below/)).toBeInTheDocument();
    // the chain is over: the key is clickable again, and master was never double-fired
    expect(screen.getByRole("button", { name: "Build the .m4b" })).toBeEnabled();
    expect(postCount(server, "/master")).toBe(1);
  });

  it("a failed job stops the finish chain with the reason instead of running master anyway", async () => {
    const user = userEvent.setup();
    const server = mountRendered(renderedCard())
      .get("/api/books/demo/render", makeSummary())
      .post("/api/books/demo/assemble", makeJob({ job_id: "j-asm", kind: "assemble", state: "queued" }));
    const { queryClient } = renderWithProviders(<RenderJobs />);

    const finishBtn = await screen.findByRole("button", { name: "Build the .m4b" });
    await waitFor(() => expect(finishBtn).toBeEnabled());
    await user.click(screen.getByRole("checkbox", { name: "also make chapter MP3s (optional)" }));
    await user.click(finishBtn);
    await waitFor(() => expect(postCount(server, "/assemble")).toBe(1));

    server.get("/api/jobs", {
      jobs: [makeJob({ job_id: "j-asm", kind: "assemble", state: "failed", is_terminal: true, error: "ffmpeg exploded" })],
    });
    void queryClient.invalidateQueries({ queryKey: ["jobs"] });

    expect(await screen.findByText(/stopped — Building the chapter MP3s failed: ffmpeg exploded/)).toBeInTheDocument();
    expect(postCount(server, "/master")).toBe(0);
  });

  /* ---- cover art (Outputs panel) ------------------------------------------------------ */

  const jpeg = () => new File([new Uint8Array([0xff, 0xd8, 0xff, 0xe0])], "cover.jpg", { type: "image/jpeg" });
  const coverInput = (container: HTMLElement) =>
    container.querySelector<HTMLInputElement>('input[type="file"][accept="image/jpeg,image/png"]');
  const withCover = (card: BookCard): BookDetail => ({
    ...makeDetail(card),
    cover: { content_type: "image/jpeg", bytes: 4096 },
  });

  it("uploading a cover PUTs multipart to /cover, refetches the book detail, and shows a cache-busted preview", async () => {
    const user = userEvent.setup();
    const card = readyCard();
    const server = mountRoutes(card)
      .get("/api/books/demo/cost-estimate", makeEstimate())
      .put("/api/books/demo/cover", { content_type: "image/jpeg", bytes: 4096 });
    const { container } = renderWithProviders(<RenderJobs />);

    // no cover yet: the control offers the first upload and shows no preview
    expect(await screen.findByRole("button", { name: "upload cover" })).toBeInTheDocument();
    expect(screen.queryByAltText("cover art")).not.toBeInTheDocument();

    // the preview may only appear via the post-upload invalidation refetching the book
    // detail — re-register the route so the refetch (and nothing else) carries the cover
    server.get("/api/books/demo", withCover(card));
    await user.upload(coverInput(container)!, jpeg());

    const img = await screen.findByAltText("cover art");
    expect(img.getAttribute("src")).toMatch(/^\/api\/books\/demo\/cover\?v=\d+$/);
    const form = server.formBodyOf("PUT", "/cover");
    expect((form.get("file") as File).name).toBe("cover.jpg");
    expect(screen.getByRole("button", { name: "replace" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "remove" })).toBeInTheDocument();
  });

  it("a 409 conflicting_job on cover upload explains the running master job in a refusal", async () => {
    const user = userEvent.setup();
    const card = readyCard();
    const server = mockApi()
      .get("/api/books", { books: [card] })
      .get("/api/books/demo", withCover(card))
      .get("/api/system", makeSystem())
      .get("/api/books/demo/cost-estimate", makeEstimate());
    server.error(
      "PUT",
      "/api/books/demo/cover",
      409,
      "conflicting_job",
      "a master job for 'demo' is running; it reads the cover mid-run",
      { job_id: "m1", book_id: "demo", kind: "master", state: "running" },
    );
    const { container } = renderWithProviders(<RenderJobs />);

    await screen.findByAltText("cover art");
    await user.upload(coverInput(container)!, jpeg());

    // plain words with the recourse, never the raw code as the main text
    expect(await screen.findByText(/Building the .m4b audiobook is still running for this book/)).toBeInTheDocument();
    expect(screen.queryByText("conflicting_job")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop building the audiobook" })).toBeInTheDocument();
  });

  it("a 415 on cover upload shows the envelope message inline", async () => {
    const user = userEvent.setup();
    const card = readyCard();
    const server = mountRoutes(card).get("/api/books/demo/cost-estimate", makeEstimate());
    server.error(
      "PUT",
      "/api/books/demo/cover",
      415,
      "unsupported_media_type",
      "cover must be image/jpeg or image/png, got image/webp",
    );
    const { container } = renderWithProviders(<RenderJobs />);

    await screen.findByRole("button", { name: "upload cover" });
    await user.upload(coverInput(container)!, jpeg());

    expect(
      await screen.findByText("cover must be image/jpeg or image/png, got image/webp"),
    ).toBeInTheDocument();
  });

  it("removing the cover takes an explicit confirm, DELETEs, and clears the preview", async () => {
    const user = userEvent.setup();
    const card = readyCard();
    const server = mockApi()
      .get("/api/books", { books: [card] })
      .get("/api/books/demo", withCover(card))
      .get("/api/system", makeSystem())
      .get("/api/books/demo/cost-estimate", makeEstimate())
      .delete("/api/books/demo/cover", null, 204);
    renderWithProviders(<RenderJobs />);

    await screen.findByAltText("cover art");
    await user.click(screen.getByRole("button", { name: "remove" }));
    // backing out deletes nothing
    await user.click(screen.getByRole("button", { name: "keep" }));
    expect(server.lastCall("DELETE", "/cover")).toBeUndefined();

    // the refetch after the delete returns a coverless detail
    server.get("/api/books/demo", makeDetail(card));
    await user.click(screen.getByRole("button", { name: "remove" }));
    await user.click(screen.getByRole("button", { name: "really remove" }));

    await waitFor(() => expect(screen.queryByAltText("cover art")).not.toBeInTheDocument());
    expect(server.lastCall("DELETE", "/cover")).toBeDefined();
    expect(await screen.findByRole("button", { name: "upload cover" })).toBeInTheDocument();
  });

  it("shows the book's jobs with running progress and failure reasons, and renders the whisper validation report", async () => {
    const card = makeBook({ attributed: true, assigned: true, rendered: true });
    const server = mockApi()
      .get("/api/books", { books: [card] })
      .get("/api/books/demo", makeDetail(card))
      .get("/api/system", makeSystem())
      .get("/api/books/demo/cost-estimate", makeEstimate());
    server.get(/\/api\/jobs\?book_id=demo/, {
      jobs: [
        makeJob({ job_id: "j-run", state: "running", progress_text: "ch 2/5 · seg 40/900" }),
        makeJob({ job_id: "j-fail", state: "failed", error: "CUDA out of memory", progress_text: "" }),
      ],
    });
    const row: ValidationRow = {
      chapter_index: 1,
      block_id: "ch001_b0042",
      segment_index: 0,
      voice_id: "v-kira",
      ok: false,
      score: 0.42,
      expected: "He drew the blade.",
      transcript: "He threw the blade.",
      synth_attempts: 2,
      reason: "'drew' heard as 'threw'",
    };
    server.get("/api/books/demo/validation", {
      validated_segments: 12,
      validation_failures: 1,
      results: [row],
    });
    server.get("/api/books/demo/render", makeSummary({ validation_failures: 1 }));
    renderWithProviders(<RenderJobs />);

    expect(await screen.findByText("running")).toBeInTheDocument();
    expect(screen.getByText("ch 2/5 · seg 40/900")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText("CUDA out of memory")).toBeInTheDocument();

    expect(await screen.findByText("1 failed of 12 checked")).toBeInTheDocument();
    expect(screen.getByText("He drew the blade.")).toBeInTheDocument();
    expect(screen.getByText("He threw the blade.")).toBeInTheDocument();
    expect(screen.getByText("'drew' heard as 'threw'")).toBeInTheDocument(); // the plain-words why
  });

  it("assembled-but-not-rendered (the post-reclone signature) shows the stale-audio banner", async () => {
    // a voice re-clone drops the manifests (rendered flips false) but deliberately keeps
    // the assembled mp3s/m4b — this banner is the loud-staleness half of that decision
    mountRoutes(makeBook({ assembled: true, mastered: true }));
    renderWithProviders(<RenderJobs />);
    expect(await screen.findByText("audio predates a voice change")).toBeInTheDocument();
  });

  /* ---- 1.12: whole-book confirm BEFORE the request (one sheet, free or paid) ------------- */

  describe("whole-book confirm before the request", () => {
    // makeDetail: 5 chapters × 30 speakable paragraphs = 150, over this limit
    const limited = (limit = 100) => makeSystem({ limits: { full_render_confirm_blocks: limit } });

    it("free: the sheet opens before anything is sent, and its key sends ONE render with confirm_full", async () => {
      const user = userEvent.setup();
      const server = mountRoutes(readyCard(), limited())
        .get("/api/books/demo/cost-estimate", makeEstimate())
        .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
      renderWithProviders(<RenderJobs />);

      await user.click(await screen.findByRole("button", { name: "record — free, nothing to approve" }));
      const dialog = await screen.findByRole("dialog", { name: "Record the whole book" });
      expect(dialog).toHaveTextContent("150");
      expect(dialog).toHaveTextContent("2.0 h"); // the book detail's runtime estimate
      expect(postCount(server, "/render")).toBe(0); // asked BEFORE the request — no 409 round trip

      await user.click(within(dialog).getByRole("button", { name: "record the whole book" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      expect(postCount(server, "/render")).toBe(1);
      expect(server.jsonBodyOf("POST", "/render")).toEqual({
        mode: "multivoice",
        chapters: [],
        apply_emotion: false,
        confirm_full: true,
      });
    });

    it("free: 'not now' sends nothing", async () => {
      const user = userEvent.setup();
      const server = mountRoutes(readyCard(), limited()).get("/api/books/demo/cost-estimate", makeEstimate());
      renderWithProviders(<RenderJobs />);

      await user.click(await screen.findByRole("button", { name: "record — free, nothing to approve" }));
      await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "not now" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      expect(postCount(server, "/render")).toBe(0);
    });

    it("under the limit it records in one click with no sheet", async () => {
      const user = userEvent.setup();
      const server = mountRoutes(readyCard(), limited(1000))
        .get("/api/books/demo/cost-estimate", makeEstimate())
        .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
      renderWithProviders(<RenderJobs />);

      await user.click(await screen.findByRole("button", { name: "record — free, nothing to approve" }));
      await waitFor(() => expect(postCount(server, "/render")).toBe(1));
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
      expect(server.jsonBodyOf("POST", "/render")).not.toHaveProperty("confirm_full");
    });

    it("a chapter range over a big book never asks", async () => {
      const user = userEvent.setup();
      const server = mountRoutes(readyCard(), limited())
        .get("/api/books/demo/cost-estimate", makeEstimate())
        .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
      renderWithProviders(<RenderJobs />);

      await user.click(await screen.findByRole("button", { name: "chapter range" }));
      await waitFor(() => expect(server.lastCall("GET", "/cost-estimate")?.url).toContain("chapters=1"));
      await user.click(await screen.findByRole("button", { name: "record — free, nothing to approve" }));
      await waitFor(() => expect(postCount(server, "/render")).toBe(1));
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("paid: the price ticket opens INSIDE the one sheet; its confirm sends the token and confirm_full together", async () => {
      const user = userEvent.setup();
      const server = mountRoutes(readyCard(), limited())
        .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7 }))
        .post("/api/books/demo/quotes", makeQuote({ token: "tok-whole" }))
        .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
      renderWithProviders(<RenderJobs />);

      await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
      const dialog = await screen.findByRole("dialog", { name: "Record the whole book" });
      // one sheet: the hours, the price and the countdown, with the ticket's key as the only confirm
      expect(dialog).toHaveTextContent("150");
      expect(within(dialog).getByText("$1.25")).toBeInTheDocument();
      expect(within(dialog).getByText(/expires \d+:\d\d/)).toBeInTheDocument();
      expect(within(dialog).queryByRole("button", { name: "record the whole book" })).not.toBeInTheDocument();
      // never two live confirm keys on screen
      expect(screen.getAllByRole("button", { name: "CONFIRM & RENDER" })).toHaveLength(1);
      expect(postCount(server, "/render")).toBe(0); // paid never auto-confirms

      await user.click(within(dialog).getByRole("button", { name: "CONFIRM & RENDER" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      expect(postCount(server, "/render")).toBe(1);
      expect(server.jsonBodyOf("POST", "/render")).toEqual({
        mode: "multivoice",
        chapters: [],
        apply_emotion: false,
        cost_token: "tok-whole",
        confirm_full: true,
      });
    });

    it("paid: closing the sheet keeps the ticket; its inline confirm reopens the sheet instead of sending", async () => {
      const user = userEvent.setup();
      const server = mountRoutes(readyCard(), limited())
        .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7 }))
        .post("/api/books/demo/quotes", makeQuote());
      renderWithProviders(<RenderJobs />);

      await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
      await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "not now" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

      await user.click(screen.getByRole("button", { name: "CONFIRM & RENDER" }));
      expect(await screen.findByRole("dialog", { name: "Record the whole book" })).toBeInTheDocument();
      expect(postCount(server, "/render")).toBe(0);
    });

    it("free: a refusal shows INSIDE the sheet (not under it), and the key is dead while sending", async () => {
      const user = userEvent.setup();
      let answer: (r: Response) => void = () => {};
      const server = mountRoutes(readyCard(), limited()).get("/api/books/demo/cost-estimate", makeEstimate());
      server.on("POST", "/api/books/demo/render", () => new Promise<Response>((res) => (answer = res)));
      renderWithProviders(<RenderJobs />);

      await user.click(await screen.findByRole("button", { name: "record — free, nothing to approve" }));
      const dialog = await screen.findByRole("dialog", { name: "Record the whole book" });
      await user.click(within(dialog).getByRole("button", { name: "record the whole book" }));
      // in flight: a second click can't send a second long job
      const sending = await within(dialog).findByRole("button", { name: "starting…" });
      expect(sending).toBeDisabled();
      await user.click(sending);
      expect(postCount(server, "/render")).toBe(1);

      answer(errorResponse(409, "conflicting_job", "another job is already working on this book"));
      expect(await within(dialog).findByText(/Another job is still running for this book/)).toBeInTheDocument();
      expect(within(dialog).queryByText("another job is already working on this book")).not.toBeInTheDocument();
      expect(within(dialog).getByRole("button", { name: "record the whole book" })).toBeEnabled();
      expect(postCount(server, "/render")).toBe(1);
    });

    it("paid: an expired ticket is re-priced inside the sheet and the free key never appears", async () => {
      const user = userEvent.setup();
      let minted = 0;
      let answerMint: (r: Response) => void = () => {};
      const server = mountRoutes(readyCard(), limited()).get(
        "/api/books/demo/cost-estimate",
        makeEstimate({ total_usd: 1.25, paid_segments: 7 }),
      );
      server.on("POST", "/api/books/demo/quotes", () => {
        minted += 1;
        return minted === 1
          ? jsonResponse(makeQuote({ token: "tok-1" }))
          : new Promise<Response>((res) => (answerMint = res));
      });
      server.on("POST", "/api/books/demo/render", () => errorResponse(409, "quote_expired", "the ticket lapsed"));
      renderWithProviders(<RenderJobs />);

      await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
      const dialog = await screen.findByRole("dialog", { name: "Record the whole book" });
      // the ticket names the book by its title, not its id
      expect(within(dialog).getByText("Demo Book")).toBeInTheDocument();
      await user.click(within(dialog).getByRole("button", { name: "CONFIRM & RENDER" }));

      // re-pricing: a placeholder, never the free key
      expect(await within(dialog).findByText("getting a fresh price…")).toBeInTheDocument();
      expect(within(dialog).queryByRole("button", { name: "record the whole book" })).not.toBeInTheDocument();

      // the re-price fails: the reason and a way to try again — still no free key
      answerMint(errorResponse(503, "missing_key", "the ElevenLabs key isn't set"));
      expect(await within(dialog).findByText(/the ElevenLabs key isn't set/)).toBeInTheDocument();
      expect(within(dialog).getByRole("button", { name: "get a fresh price" })).toBeInTheDocument();
      expect(within(dialog).queryByRole("button", { name: "record the whole book" })).not.toBeInTheDocument();
      expect(postCount(server, "/render")).toBe(1);
    });
  });

  /* ---- 1.12: the quote never outlives the setup it priced --------------------------------- */

  it("the live quote is cleared on ANY scope, mode, emotion or force change", async () => {
    const user = userEvent.setup();
    const card = makeBook({ attributed: true, assigned: true, rendered: true });
    const server = mountRoutes(card)
      .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7 }))
      .get("/api/books/demo/render", makeSummary())
      .get("/api/books/demo/validation", { validated_segments: 0, validation_failures: 0, results: [] })
      .post("/api/books/demo/quotes", makeQuote());
    renderWithProviders(<RenderJobs />);

    const mintLive = async () => {
      await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
      expect(await screen.findByRole("button", { name: "CONFIRM & RENDER" })).toBeInTheDocument();
    };
    const gone = () =>
      waitFor(() => expect(screen.queryByRole("button", { name: "CONFIRM & RENDER" })).not.toBeInTheDocument());

    await mintLive();
    await user.click(screen.getByRole("checkbox", { name: "voice per-line emotion" }));
    await gone();

    await mintLive();
    await user.click(screen.getByRole("button", { name: "chapter range" }));
    await gone();

    await mintLive();
    await user.click(screen.getByRole("checkbox", { name: "record again from scratch — ignore saved audio" }));
    await gone();

    await mintLive();
    await user.click(screen.getByRole("button", { name: "One narrator" }));
    await gone();
    expect(postCount(server, "/render")).toBe(0);
    expect(postCount(server, "/quotes")).toBe(4);

    // One narrator: the narrator and the speed are part of what was priced too
    await waitFor(() => expect(server.lastCall("GET", "/cost-estimate")?.url).toContain("mode=single"));
    await mintLive();
    await user.click(screen.getByRole("button", { name: /narrator voice/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: /^Adam/ }));
    await gone();

    await mintLive();
    await user.click(screen.getByRole("button", { name: /narrator speed/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "a little faster" }));
    await gone();
    expect(postCount(server, "/render")).toBe(0);
  });

  it("a price that lands AFTER a setting changed is dropped — it priced the old setup", async () => {
    const user = userEvent.setup();
    let answer: (r: Response) => void = () => {};
    const server = mountRoutes(readyCard()).get(
      "/api/books/demo/cost-estimate",
      makeEstimate({ total_usd: 1.25, paid_segments: 7 }),
    );
    server.on("POST", "/api/books/demo/quotes", () => new Promise<Response>((res) => (answer = res)));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    await waitFor(() => expect(postCount(server, "/quotes")).toBe(1));
    await user.click(screen.getByRole("checkbox", { name: "voice per-line emotion" }));
    answer(jsonResponse(makeQuote()));

    await waitFor(() => expect(screen.queryByText("getting the price…")).not.toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "CONFIRM & RENDER" })).not.toBeInTheDocument();
    expect(postCount(server, "/render")).toBe(0);
  });

  /* ---- 1.12: One narrator — voice and speed ------------------------------------------------ */

  it("One narrator: a built-in voice and speed go into the estimate, the quote and the render alike", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard())
      .get("/api/books/demo/cost-estimate", makeEstimate({ total_usd: 1.25, paid_segments: 7, mode: "single" }))
      .post("/api/books/demo/quotes", makeQuote({ token: "tok-n" }))
      .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: "One narrator" }));
    // the standard narrator is named from settings + the preset catalog
    expect(await screen.findByRole("button", { name: /Standard narrator \(Heart\)/ })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /narrator voice/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: /^Adam/ }));
    await user.click(screen.getByRole("button", { name: /narrator speed/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "a little faster" }));

    await waitFor(() => {
      const url = server.lastCall("GET", "/cost-estimate")?.url ?? "";
      expect(url).toContain("mode=single");
      expect(url).toContain("engine=kokoro&voice=am_adam&speed=1.1");
    });
    await user.click(await screen.findByRole("button", { name: /get the price — about \$1\.25/ }));
    await user.click(await screen.findByRole("button", { name: "CONFIRM & RENDER" }));
    await waitFor(() => expect(postCount(server, "/render")).toBe(1));
    const spec = { engine: "kokoro", voice: "am_adam", speed: 1.1 };
    expect(server.jsonBodyOf("POST", "/quotes")).toEqual({ mode: "single", chapters: [], single: spec });
    expect(server.jsonBodyOf("POST", "/render")).toEqual({
      mode: "single",
      chapters: [],
      single: spec,
      cost_token: "tok-n",
    });
  });

  it("One narrator: a saved library voice records with its own settings — no speed is sent", async () => {
    const user = userEvent.setup();
    const server = mountRoutes(readyCard())
      .get("/api/books/demo/cost-estimate", makeEstimate({ mode: "single" }))
      .post("/api/books/demo/render", makeJob({ kind: "render", state: "queued" }));
    renderWithProviders(<RenderJobs />);

    await user.click(await screen.findByRole("button", { name: "One narrator" }));
    await user.click(await screen.findByRole("button", { name: /narrator voice/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: /^Kira · your voice/ }));
    expect(await screen.findByText("this voice reads at its own saved speed")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /narrator speed/ })).not.toBeInTheDocument();

    await waitFor(() =>
      expect(server.lastCall("GET", "/cost-estimate")?.url).toContain("engine=chatterbox&voice=v-kira"),
    );
    await user.click(await screen.findByRole("button", { name: "record — free, nothing to approve" }));
    await waitFor(() => expect(postCount(server, "/render")).toBe(1));
    expect(server.jsonBodyOf("POST", "/render")).toEqual({
      mode: "single",
      chapters: [],
      single: { engine: "chatterbox", voice: "v-kira" },
    });
  });

  it("One narrator over a chapter range warns when the other chapters had a different narrator", async () => {
    const user = userEvent.setup();
    mountRoutes(makeBook({ rendered: true }))
      .get("/api/books/demo/cost-estimate", makeEstimate({ mode: "single" }))
      .get("/api/books/demo/validation", { validated_segments: 0, validation_failures: 0, results: [] })
      .get(
        "/api/books/demo/render",
        makeSummary({ mode: "single", voice_id: "am_adam", active_mode: "single", available_modes: ["single"] }),
      );
    renderWithProviders(<RenderJobs />);

    // the recorded narrator is preselected, so a range re-make keeps it — no warning
    expect(await screen.findByRole("button", { name: /Adam.*narrator voice|narrator voice.*Adam/ })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "chapter range" }));
    expect(screen.queryByText("narrator changed")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /narrator voice/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: /Standard narrator/ }));
    expect(await screen.findByText("narrator changed")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "record the whole book instead" }));
    await waitFor(() => expect(screen.queryByText("narrator changed")).not.toBeInTheDocument());
  });

  it("One narrator waits for the voice lists before pricing, so the earlier narrator isn't swapped for the default", async () => {
    let answer: (r: Response) => void = () => {};
    const server = mountRoutes(makeBook({ rendered: true }))
      .get("/api/books/demo/cost-estimate", makeEstimate({ mode: "single" }))
      .get("/api/books/demo/validation", { validated_segments: 0, validation_failures: 0, results: [] })
      .get(
        "/api/books/demo/render",
        makeSummary({ mode: "single", voice_id: "am_adam", active_mode: "single", available_modes: ["single"] }),
      );
    server.on("GET", "/api/engines/kokoro/voices", () => new Promise<Response>((res) => (answer = res)));
    renderWithProviders(<RenderJobs />);

    expect(await screen.findByText("finding the narrator of the earlier recording…")).toBeInTheDocument();
    expect(server.lastCall("GET", "/cost-estimate")).toBeUndefined();
    expect(screen.queryByRole("button", { name: "record — free, nothing to approve" })).not.toBeInTheDocument();

    answer(
      jsonResponse({
        engine_id: "kokoro",
        voices: [{ id: "am_adam", name: "Adam", language: "en-us", gender: "male", description: null }],
      }),
    );
    await waitFor(() =>
      expect(server.lastCall("GET", "/cost-estimate")?.url).toContain("engine=kokoro&voice=am_adam"),
    );
    expect(server.calls.filter((c) => c.url.includes("/cost-estimate")).every((c) => c.url.includes("am_adam"))).toBe(
      true,
    );
  });

  it("One narrator says so when the earlier narrator is no longer in the voices", async () => {
    mountRoutes(makeBook({ rendered: true }))
      .get("/api/books/demo/cost-estimate", makeEstimate({ mode: "single" }))
      .get("/api/books/demo/validation", { validated_segments: 0, validation_failures: 0, results: [] })
      .get(
        "/api/books/demo/render",
        makeSummary({ mode: "single", voice_id: "v-gone", active_mode: "single", available_modes: ["single"] }),
      );
    renderWithProviders(<RenderJobs />);
    expect(await screen.findByText("narrator not found")).toBeInTheDocument();
  });

  /* ---- 1.12: cast changed + range, Kokoro reads flat --------------------------------------- */

  it("after a recast, a chapter range is steered to the whole book", async () => {
    const user = userEvent.setup();
    mountRendered(renderedCard()).get("/api/books/demo/render", makeSummary({ rendered_assignment_hash: "ah-OLD" }));
    renderWithProviders(<RenderJobs />);

    await screen.findByText(/the current cast differs from the recorded audio/);
    await user.click(screen.getByRole("button", { name: "chapter range" }));
    expect(await screen.findByText(/A chapter range can't use the new cast/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "record the whole book instead" }));
    expect(await screen.findByText(/the current cast differs from the recorded audio/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "whole book" })).toHaveClass("on");
  });

  it("the emotion option says Kokoro voices read flat", async () => {
    mountRoutes(readyCard()).get("/api/books/demo/cost-estimate", makeEstimate());
    renderWithProviders(<RenderJobs />);
    expect(await screen.findByText(/Kokoro voices always read flat/)).toBeInTheDocument();
  });

  /* ---- 1.12: audio-check rows lead to the fix ---------------------------------------------- */

  it("an audio-check row links to Review and Pronunciation, and Re-make sets up a forced chapter without sending", async () => {
    const user = userEvent.setup();
    const row: ValidationRow = {
      chapter_index: 3,
      block_id: "ch003_b0012",
      segment_index: 1,
      voice_id: "v-kira",
      ok: false,
      score: 0.61,
      expected: "Kael'thas raised a hand.",
      transcript: "Kale has raised a hand.",
      synth_attempts: 3,
      reason: "dropped or changed words",
    };
    const server = mountRoutes(renderedCard())
      .get("/api/books/demo/cost-estimate", makeEstimate())
      .get("/api/books/demo/render", makeSummary())
      .get("/api/books/demo/validation", { validated_segments: 9, validation_failures: 1, results: [row] });
    renderWithProviders(<RenderJobs />);

    const fixes = within(await screen.findByRole("group", { name: "fix this line" }));
    expect(fixes.getByRole("link", { name: "Open in Review" })).toHaveAttribute(
      "href",
      "/review?book=demo&ch=3&block=ch003_b0012&seg=1",
    );
    expect(fixes.getByRole("link", { name: "Fix pronunciation of “Kael'thas”" })).toHaveAttribute(
      "href",
      "/lexicon?book=demo&term=Kael%27thas",
    );
    // the voice reads by name, not id
    expect(await screen.findByText("· Kira")).toBeInTheDocument();

    await user.click(fixes.getByRole("button", { name: "Re-make chapter 3" }));
    const dialog = await screen.findByRole("dialog", { name: "Re-make chapter 3" });
    // makeSummary has only chapter 1's recorded lines, so chapter 3 falls back to its paragraphs
    expect(dialog).toHaveTextContent("30 paragraphs");
    expect(dialog).toHaveTextContent(/paid voice are billed again/);
    await user.click(within(dialog).getByRole("button", { name: "set up the re-make" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("checkbox", { name: "record again from scratch — ignore saved audio" })).toBeChecked();
    expect(screen.getByRole("spinbutton", { name: "ch" })).toHaveValue(3);
    expect(screen.getByRole("spinbutton", { name: "to" })).toHaveValue(3);
    await waitFor(() => {
      const url = server.lastCall("GET", "/cost-estimate")?.url ?? "";
      expect(url).toContain("chapters=3");
      expect(url).toContain("force=true");
    });
    // keyboard focus lands on the recording controls, not back on the row's Re-make button
    await waitFor(() => expect(screen.getByRole("group", { name: "how it sounds" })).toHaveFocus());
    // never auto-confirms: it set things up, nothing was sent
    expect(postCount(server, "/render")).toBe(0);
    expect(postCount(server, "/quotes")).toBe(0);
  });

  describe("next step", () => {
    it("with no books the empty state's one action is adding a book", async () => {
      mockApi().get("/api/books", { books: [] });
      renderWithProviders(<RenderJobs />);
      expect(await screen.findByRole("link", { name: "Add a book" })).toHaveAttribute("href", "/#add");
    });

    it("the rail's button lands on the attribute box on this page and scrolls it into view", async () => {
      const user = userEvent.setup();
      mountRoutes(makeBook()).get("/api/books/demo/cost-estimate", makeEstimate({ mode: "single" }));
      const scrolled: Element[] = [];
      const spy = vi.spyOn(Element.prototype, "scrollIntoView").mockImplementation(function (this: Element) {
        scrolled.push(this);
      });
      try {
        renderWithProviders(<RenderJobs />, { route: "/render?book=demo" });
        const rail = await screen.findByRole("group", { name: "pipeline" });
        const next = await within(rail).findByRole("link", { name: "Find the speakers" });
        expect(next).toHaveAttribute("href", "/render?book=demo#find-speakers");
        expect(screen.queryByText(/^next:/)).not.toBeInTheDocument();

        await user.click(next);
        await waitFor(() => expect(scrolled.some((el) => el.id === "find-speakers")).toBe(true));
        // the box it lands on is the one that starts it, named the same way, and keyboard
        // focus moves into it so the next Tab reaches its button
        const box = document.getElementById("find-speakers")!;
        expect(within(box).getByRole("button", { name: /find the speakers in/ })).toBeInTheDocument();
        expect(within(box).getByText("Full cast: find the speakers")).toBeInTheDocument();
        expect(within(box).queryByText(/stage_prerequisite|speakers not found yet/)).not.toBeInTheDocument();
        await waitFor(() => expect(document.activeElement).toBe(box));
      } finally {
        spy.mockRestore();
      }
    });

    it("a rendered book's step is Build the .m4b, anchored on the Outputs panel", async () => {
      mountRoutes(makeBook({ attributed: true, assigned: true, rendered: true }))
        .get("/api/books/demo/cost-estimate", makeEstimate())
        .get("/api/books/demo/render", makeSummary())
        .get("/api/books/demo/validation", { validated_segments: 0, validation_failures: 0, results: [] });
      renderWithProviders(<RenderJobs />, { route: "/render?book=demo" });
      const rail = await screen.findByRole("group", { name: "pipeline" });
      expect(await within(rail).findByRole("link", { name: "Build the .m4b" })).toHaveAttribute(
        "href",
        "/render?book=demo#build",
      );
      expect(document.getElementById("build")).toHaveTextContent("Outputs");
    });
  });
});
