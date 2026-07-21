import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type {
  BookCard,
  BookDetail,
  CostEstimateOut,
  QuoteResponse,
  RenderSummaryOut,
  SystemStatusOut,
  ValidationRow,
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
    .get("/api/system", system);
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
      await screen.findByText(/7 segment\(s\) use a paid cloud voice — about \$1\.25\./),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /approve — mint quote for \$1\.25/ })).toBeInTheDocument();
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
    expect(await screen.findByText(/system default: on/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "voice per-line emotion" })).toBeChecked();
    await waitFor(() =>
      expect(server.lastCall("GET", "/cost-estimate")?.url).toContain("apply_emotion=true"),
    );

    // mint WITHOUT touching the toggle: the server default must reach the quote fingerprint
    await user.click(await screen.findByRole("button", { name: /approve — mint quote for \$1\.25/ }));
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

    await user.click(await screen.findByRole("button", { name: "single voice" }));
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

    await user.click(await screen.findByRole("button", { name: /approve — mint quote for \$1\.25/ }));
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
      await screen.findByRole("button", { name: "render multivoice — free, nothing to approve" }),
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

    await user.click(await screen.findByRole("button", { name: /approve — mint quote for \$1\.25/ }));

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

    await user.click(await screen.findByRole("button", { name: /approve — mint quote for \$1\.25/ }));
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

    await user.click(await screen.findByRole("button", { name: /approve — mint quote for \$1\.25/ }));
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

    await user.click(await screen.findByRole("button", { name: /approve — mint quote for \$1\.25/ }));
    await user.click(await screen.findByRole("button", { name: "CONFIRM & RENDER" }));

    await waitFor(() => expect(postCount(server, "/quotes")).toBe(2));
    // the fresh ticket is live again (not stamped) and still waits for an explicit confirm
    expect(await screen.findByRole("button", { name: "CONFIRM & RENDER" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "RE-MINT QUOTE" })).not.toBeInTheDocument();
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

    await user.click(await screen.findByRole("button", { name: /approve — mint quote for \$1\.25/ }));
    await user.click(await screen.findByRole("button", { name: "CONFIRM & RENDER" }));

    expect(await screen.findByText("DRIFT")).toBeInTheDocument();
    expect(screen.getByText("the cache shifted under this quote")).toBeInTheDocument();
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
      await screen.findByRole("button", { name: "render multivoice — free, nothing to approve" }),
    );

    const dialog = await screen.findByRole("dialog", { name: "Full-book render" });
    expect(dialog).toHaveTextContent(/4,?200/); // the refusal detail (segment count) is shown
    await user.click(screen.getByRole("button", { name: "render the whole book" }));

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

    await user.click(await screen.findByRole("button", { name: /approve — mint quote for \$1\.25/ }));
    await user.click(await screen.findByRole("button", { name: "CONFIRM & RENDER" }));

    // the whole-book gate fires ON TOP of the paid flow — confirming must not drop the ticket
    const dialog = await screen.findByRole("dialog", { name: "Full-book render" });
    await user.click(within(dialog).getByRole("button", { name: "render the whole book" }));

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
    await screen.findByText(/anthropic · claude-sonnet-4/);
    await user.click(screen.getByRole("button", { name: /attribute the whole book/ }));

    // the 402 surfaces the reason and an explicit confirm control — nothing auto-retries
    expect(await screen.findByText(/anthropic attribution is a paid run/)).toBeInTheDocument();
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

  /* ---- active render mode (instant fallback) ------------------------------------------ */

  const modeGroup = async () => within(await screen.findByRole("group", { name: "active render mode" }));
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
    expect(group.getByRole("button", { name: "multivoice · active" })).toBeDisabled();
    const switchBtn = group.getByRole("button", { name: "single voice · switch" });
    expect(switchBtn).toBeEnabled();

    // the flipped chip may only appear via the post-switch invalidation refetching the
    // summary — re-register the GET so the refetch (and nothing else) carries it
    server.get(
      "/api/books/demo/render",
      makeSummary({ mode: "single", active_mode: "single", available_modes: ["single", "multi"] }),
    );
    await user.click(switchBtn);

    expect(await group.findByRole("button", { name: "single voice · active" })).toBeDisabled();
    expect(group.getByRole("button", { name: "multivoice · switch" })).toBeEnabled();
    expect(server.jsonBodyOf("POST", "/render/mode")).toEqual({ mode: "single" });
  });

  it("the mode control never mounts before the first render", async () => {
    const server = mountRoutes(readyCard()).get("/api/books/demo/cost-estimate", makeEstimate());
    renderWithProviders(<RenderJobs />);

    await screen.findByRole("button", { name: "render multivoice — free, nothing to approve" });
    expect(screen.queryByRole("group", { name: "active render mode" })).not.toBeInTheDocument();
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

    await user.click((await modeGroup()).getByRole("button", { name: "single voice · switch" }));

    expect(await screen.findByText("conflicting_job")).toBeInTheDocument();
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
    expect(group.getByRole("button", { name: "single voice · active" })).toBeDisabled();
    expect(group.getByRole("button", { name: "multivoice · not rendered" })).toBeDisabled();
    expect(screen.getByText(/render the other mode once and you can switch/)).toBeInTheDocument();
  });

  it("while a render job runs both chips deaden and the hint says the job owns the render", async () => {
    const card = {
      ...renderedCard(),
      active_job: { job_id: "j-1", kind: "render" as const, state: "running" as const },
    };
    mountRendered(card).get("/api/books/demo/render", makeSummary({ available_modes: ["single", "multi"] }));
    renderWithProviders(<RenderJobs />);

    const group = await modeGroup();
    expect(group.getByRole("button", { name: "single voice · switch" })).toBeDisabled();
    expect(
      screen.getByText(/a render job is running and owns the render — wait for it or cancel it/),
    ).toBeInTheDocument();
  });

  /* ---- re-render (force) + stale-cast banner ------------------------------------------ */

  const forceToggle = () =>
    screen.findByRole("checkbox", { name: "force re-render — ignore cached audio" });

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
      await screen.findByRole("button", { name: "render multivoice — free, nothing to approve" }),
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
    await screen.findByRole("button", { name: "render multivoice — free, nothing to approve" });
    expect(
      screen.queryByRole("checkbox", { name: "force re-render — ignore cached audio" }),
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
      await screen.findByText(/the current cast differs from the rendered audio/),
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
      screen.queryByText(/the current cast differs from the rendered audio/),
    ).not.toBeInTheDocument();
  });

  /* ---- one-click finish (assemble → master) ------------------------------------------- */

  it("finish runs assemble, waits for it to land, then runs master and reports done", async () => {
    const user = userEvent.setup();
    const server = mountRendered(renderedCard())
      .get("/api/books/demo/render", makeSummary())
      .post("/api/books/demo/assemble", makeJob({ job_id: "j-asm", kind: "assemble", state: "queued" }));
    const { queryClient } = renderWithProviders(<RenderJobs />);

    // the key deadens until the book detail (status.rendered) lands
    const finishBtn = await screen.findByRole("button", { name: "finish · mp3s + m4b" });
    await waitFor(() => expect(finishBtn).toBeEnabled());
    await user.click(finishBtn);

    // assemble fires immediately; master must NOT — it waits for assemble to succeed
    await waitFor(() => expect(postCount(server, "/assemble")).toBe(1));
    expect(postCount(server, "/master")).toBe(0);
    expect(screen.getByText(/then master — keep this page open/)).toBeInTheDocument();

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
    // the chain is over: finish is clickable again, and master was never double-fired
    expect(screen.getByRole("button", { name: "finish · mp3s + m4b" })).toBeEnabled();
    expect(postCount(server, "/master")).toBe(1);
  });

  it("a failed job stops the finish chain with the reason instead of running master anyway", async () => {
    const user = userEvent.setup();
    const server = mountRendered(renderedCard())
      .get("/api/books/demo/render", makeSummary())
      .post("/api/books/demo/assemble", makeJob({ job_id: "j-asm", kind: "assemble", state: "queued" }));
    const { queryClient } = renderWithProviders(<RenderJobs />);

    const finishBtn = await screen.findByRole("button", { name: "finish · mp3s + m4b" });
    await waitFor(() => expect(finishBtn).toBeEnabled());
    await user.click(finishBtn);
    await waitFor(() => expect(postCount(server, "/assemble")).toBe(1));

    server.get("/api/jobs", {
      jobs: [makeJob({ job_id: "j-asm", kind: "assemble", state: "failed", is_terminal: true, error: "ffmpeg exploded" })],
    });
    void queryClient.invalidateQueries({ queryKey: ["jobs"] });

    expect(await screen.findByText(/auto-finish stopped — the assemble job failed: ffmpeg exploded/)).toBeInTheDocument();
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
    );
    const { container } = renderWithProviders(<RenderJobs />);

    await screen.findByAltText("cover art");
    await user.upload(coverInput(container)!, jpeg());

    expect(await screen.findByText("conflicting_job")).toBeInTheDocument();
    expect(screen.getByText(/a master job is running and reads the cover mid-run/)).toBeInTheDocument();
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
  });

  it("assembled-but-not-rendered (the post-reclone signature) shows the stale-audio banner", async () => {
    // a voice re-clone drops the manifests (rendered flips false) but deliberately keeps
    // the assembled mp3s/m4b — this banner is the loud-staleness half of that decision
    mountRoutes(makeBook({ assembled: true, mastered: true }));
    renderWithProviders(<RenderJobs />);
    expect(await screen.findByText("audio predates a voice change")).toBeInTheDocument();
  });
});
