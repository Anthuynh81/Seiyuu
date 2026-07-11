import { fireEvent, screen, waitFor } from "@testing-library/react";
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
function mountRoutes(card: BookCard): MockApi {
  return mockApi()
    .get("/api/books", { books: [card] })
    .get("/api/books/demo", makeDetail(card))
    .get("/api/system", makeSystem());
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
    await user.click(screen.getByRole("button", { name: "attribute the whole book" }));

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
    const summary: RenderSummaryOut = {
      book_id: "demo",
      mode: "multivoice",
      chapters: [{ index: 1, title: "Chapter 1", segments: 300, duration_seconds: 1800 }],
      total_seconds: 1800,
      voices_used: {},
      validation_failures: 1,
    };
    server.get("/api/books/demo/render", summary);
    renderWithProviders(<RenderJobs />);

    expect(await screen.findByText("running")).toBeInTheDocument();
    expect(screen.getByText("ch 2/5 · seg 40/900")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText("CUDA out of memory")).toBeInTheDocument();

    expect(await screen.findByText("1 failed of 12 checked")).toBeInTheDocument();
    expect(screen.getByText("He drew the blade.")).toBeInTheDocument();
    expect(screen.getByText("He threw the blade.")).toBeInTheDocument();
  });
});
