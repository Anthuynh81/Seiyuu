import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type {
  AttributionOut,
  BookDetail,
  CharacterSummary,
  CharactersOverview,
  SegmentBrowserOut,
  SegmentRow,
  VoiceAssignment,
  VoiceOut,
} from "../api/types";
import { makeBook } from "../test/fixtures";
import type { MockApi } from "../test/utils";
import { mockApi, renderWithProviders } from "../test/utils";
import { Review } from "./Review";

/* ------------------------------------------------------------------ fixtures */

function alice(): CharacterSummary {
  return {
    id: "alice",
    name: "Alice",
    aliases: ["Ali"],
    gender: "f",
    age_hint: null,
    line_count: 42,
    sample_lines: ['"Hello there," she said.'],
    first_appearance: "ch001_b0001",
  };
}

function bob(): CharacterSummary {
  return {
    id: "bob",
    name: "Bob",
    aliases: [],
    gender: "m",
    age_hint: null,
    line_count: 17,
    sample_lines: [],
    first_appearance: "ch001_b0002",
  };
}

function overviewFor(bookId: string, characters: CharacterSummary[]): CharactersOverview {
  return {
    book_id: bookId,
    provider_id: "local",
    model_id: "qwen2.5:7b",
    prompt_version: "v6",
    narration_segments: 120,
    low_confidence_segments: 3,
    confidence_threshold: 0.7,
    characters,
    flagged: [],
    notes: [],
    edit_warnings: [],
  };
}

function detailFor(bookId: string, title: string, attributed = true): BookDetail {
  return {
    status: {
      book_id: bookId,
      title,
      authors: ["A. Author"],
      ingested: true,
      attributed,
      assigned: false,
      rendered: false,
      assembled: false,
      mastered: false,
    },
    chapters: [
      { index: 1, title: "The Beginning", blocks: 3, speakable_blocks: 3 },
      { index: 2, title: "The Middle", blocks: 3, speakable_blocks: 3 },
    ],
    runtime_estimate_seconds: null,
    active_job: null,
    recent_jobs: [],
    downloads: { m4b: null, chapter_mp3s: [] },
    cover: null,
  };
}

function seg(overrides: Partial<SegmentRow> = {}): SegmentRow {
  return {
    block_id: "ch001_b0001",
    segment_index: 0,
    type: "narration",
    speaker: null,
    speaker_name: null,
    text: "It was a dark night.",
    confidence: 0.99,
    has_audio: false,
    audio_segment: null,
    duration_seconds: null,
    voice_id: null,
    audio_key: null,
    ...overrides,
  };
}

function chapterOneSegments(): SegmentBrowserOut {
  return {
    chapter_index: 1,
    title: "The Beginning",
    segments: [
      seg(),
      seg({
        segment_index: 1,
        type: "dialogue",
        speaker: "alice",
        speaker_name: "Alice",
        text: '"Hello there," said Alice.',
        confidence: 0.55,
      }),
    ],
    edit_warnings: [],
  };
}

/** Emotion side-channel: the dialogue segment (ordinal 1 within its block) is tagged angry/2. */
function chapterOneAttribution(): AttributionOut {
  return {
    report: {
      chapters: [
        {
          index: 1,
          segments: [{ block_id: "ch001_b0001" }, { block_id: "ch001_b0001" }],
          segment_emotions: [null, { label: "angry", intensity: 2 }],
        },
      ],
    },
    edit_warnings: [],
  };
}

function voice(id: string, name: string): VoiceOut {
  return {
    voice_id: id,
    name,
    kind: "blend",
    engine: "kokoro",
    preset_id: null,
    blend: null,
    reference_audio: null,
    seed: 7,
    consent_attested: false,
    consent: null,
    tags: [],
    created_at: "2026-07-11T00:00:00Z",
    has_audition: false,
  };
}

const ASSIGNMENT: VoiceAssignment = {
  schema_version: 1,
  book_id: "b1",
  stage: "draft",
  narrator_voice_id: "v-narr",
  assignments: { alice: "v-alice", bob: "v-bob" },
  thought_voice_id: null,
  created_at: "2026-07-11T00:00:00Z",
};

function registerBook(server: MockApi, bookId: string, title: string, characters?: CharacterSummary[]) {
  server
    .get(`/api/books/${bookId}`, detailFor(bookId, title))
    .get(`/api/books/${bookId}/characters`, overviewFor(bookId, characters ?? [alice(), bob()]))
    .get(`/api/books/${bookId}/chapters/1/segments`, chapterOneSegments())
    .get(`/api/books/${bookId}/attribution`, chapterOneAttribution())
    .get(`/api/books/${bookId}/edits`, { version: 1, ops: [{ op: "rename" }] })
    .post(`/api/books/${bookId}/edits`, { ok: true })
    .error("GET", `/api/books/${bookId}/assignment`, 404, "not_found", "no assignment yet")
    .get("/api/voices", {
      voices: [voice("v-narr", "Narrator"), voice("v-alice", "auto:Alice"), voice("v-bob", "auto:Bob")],
      unreadable: [],
    });
}

/** Baseline: one attributed book b1, chapter 1 open, one prior edit, no casting yet. */
function setupReview(characters?: CharacterSummary[]): MockApi {
  const server = mockApi().get("/api/books", {
    books: [makeBook({ book_id: "b1", title: "Attributed One", attributed: true })],
  });
  registerBook(server, "b1", "Attributed One", characters);
  return server;
}

/* ------------------------------------------------------------------ tests */

describe("Review", () => {
  it("selects the book from the ?book= query param and renders its roster + review-queue stats", async () => {
    const server = mockApi().get("/api/books", {
      books: [
        makeBook({ book_id: "b1", title: "Book One", attributed: true }),
        makeBook({ book_id: "b2", title: "Book Two", attributed: true }),
      ],
    });
    const zed: CharacterSummary = { ...alice(), id: "zed", name: "Zed", aliases: [], sample_lines: [] };
    registerBook(server, "b2", "Book Two", [zed, bob()]);
    renderWithProviders(<Review />, { route: "/review?book=b2" });

    // b2's roster, not the shelf-default b1
    expect(await screen.findByText("Zed")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Book Two/ })).toBeInTheDocument();
    expect(server.lastCall("GET", "/api/books/b1/characters")).toBeUndefined();
    // review-queue stats + attribution provenance from /characters
    expect(screen.getByText(/3 low-confidence in the book · 1 in this chapter/)).toBeInTheDocument();
    expect(screen.getByText(/read by local · qwen2.5:7b · v6/)).toBeInTheDocument();
    // the chapter page renders segments with the speaker chip in the margin
    expect(screen.getByText("It was a dark night.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ALICE" })).toBeInTheDocument();
  });

  it("shows the stage_prerequisite refusal for an unattributed book and fetches no review data", async () => {
    const server = mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Raw Ingest" })] })
      .get("/api/books/b1", detailFor("b1", "Raw Ingest", false))
      .get("/api/voices", { voices: [], unreadable: [] });
    renderWithProviders(<Review />, { route: "/review?book=b1" });

    expect(await screen.findByText(/this book has no attribution yet/)).toBeInTheDocument();
    expect(server.lastCall("GET", "/characters")).toBeUndefined();
    expect(server.lastCall("GET", "/segments")).toBeUndefined();
  });

  it("masks characters that debut beyond the reading frontier until revealed", async () => {
    const user = userEvent.setup();
    const mallory: CharacterSummary = {
      ...bob(),
      id: "mallory",
      name: "Mallory",
      first_appearance: "ch005_b0010",
    };
    setupReview([alice(), bob(), mallory]);
    renderWithProviders(<Review />);

    expect(await screen.findByText("Alice")).toBeInTheDocument();
    // debut ch 5 > frontier ch 1: the name must not leak
    expect(screen.queryByText("Mallory")).not.toBeInTheDocument();
    expect(screen.getByText(/enters ch 5/)).toBeInTheDocument();

    await user.click(screen.getByRole("link", { name: "reveal" }));
    expect(await screen.findByText("Mallory")).toBeInTheDocument();
  });

  it("rename records an op=rename edit and the roster refreshes with the new name", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    renderWithProviders(<Review />);

    // scope to Alice's roster row: the tests edit ALICE, wherever the roster sorts her
    const aliceRow = await screen.findByRole("row", { name: /Alice/ });
    await user.click(within(aliceRow).getByRole("button", { name: "✎" }));
    const dialog = await screen.findByRole("dialog", { name: /Edit character — Alice/ });
    const input = within(dialog).getByRole("textbox");
    await user.clear(input);
    await user.type(input, "Alicia");

    // the refetch after the edit returns the renamed roster
    server.get("/api/books/b1/characters", overviewFor("b1", [{ ...alice(), name: "Alicia" }, bob()]));
    await user.click(within(dialog).getByRole("button", { name: "rename" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.jsonBodyOf("POST", "/api/books/b1/edits")).toEqual({
      op: "rename",
      character_id: "alice",
      new_name: "Alicia",
    });
    expect(await screen.findByText("Alicia")).toBeInTheDocument();
  });

  it("merge records an op=merge edit with this character as loser and the picked one as winner", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    renderWithProviders(<Review />);

    // scope to Alice's roster row: the tests edit ALICE, wherever the roster sorts her
    const aliceRow = await screen.findByRole("row", { name: /Alice/ });
    await user.click(within(aliceRow).getByRole("button", { name: "✎" }));
    const dialog = await screen.findByRole("dialog", { name: /Edit character — Alice/ });
    expect(within(dialog).getByRole("button", { name: "merge" })).toBeDisabled();

    await user.click(within(dialog).getByRole("button", { name: /— choose —/ }));
    await user.click(await screen.findByRole("option", { name: "Bob" }));
    await user.click(within(dialog).getByRole("button", { name: "merge" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.jsonBodyOf("POST", "/api/books/b1/edits")).toEqual({
      op: "merge",
      loser_id: "alice",
      winner_id: "bob",
    });
  });

  it("the speaker chip opens the reassign popover and records an op=reassign anchored to the segment", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "ALICE" }));
    // the speaker select starts on the current speaker; move it to Bob
    await user.click(screen.getByRole("button", { name: /Alice/ }));
    await user.click(await screen.findByRole("option", { name: "Bob" }));
    await user.click(screen.getByRole("button", { name: "record edit" }));

    await waitFor(() => expect(screen.queryByRole("button", { name: "record edit" })).not.toBeInTheDocument());
    expect(server.jsonBodyOf("POST", "/api/books/b1/edits")).toEqual({
      op: "reassign",
      block_id: "ch001_b0001",
      segment_index: 1,
      speaker: "bob",
    });
  });

  it("renders the emotion chip from the attribution side-channel; set/clear post op=set_emotion overrides", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    renderWithProviders(<Review />);

    // the side-channel tag lines up with the dialogue segment
    const chip = await screen.findByRole("button", { name: "angry ••" });
    expect(screen.getByText(/1 emotion tag\(s\) in this chapter/)).toBeInTheDocument();

    await user.click(chip);
    await user.click(await screen.findByRole("button", { name: "set emotion" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "set emotion" })).not.toBeInTheDocument());
    expect(server.jsonBodyOf("POST", "/api/books/b1/edits")).toEqual({
      op: "set_emotion",
      block_id: "ch001_b0001",
      segment_index: 1,
      emotion: { label: "angry", intensity: 2 },
    });

    // clear records the same op with a null verdict
    await user.click(screen.getByRole("button", { name: "angry ••" }));
    await user.click(await screen.findByRole("button", { name: "clear" }));
    await waitFor(() =>
      expect(server.jsonBodyOf("POST", "/api/books/b1/edits")).toEqual({
        op: "set_emotion",
        block_id: "ch001_b0001",
        segment_index: 1,
        emotion: null,
      }),
    );
  });

  it("undo last DELETEs /edits/last and the drained log disables further undo", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.delete("/api/books/b1/edits/last", { removed: { op: "rename" } });
    renderWithProviders(<Review />);

    const undoBtn = await screen.findByRole("button", { name: "undo last" });
    await waitFor(() => expect(undoBtn).toBeEnabled());

    server.get("/api/books/b1/edits", { version: 1, ops: [] }); // the refetch after the undo
    await user.click(undoBtn);

    expect(server.lastCall("DELETE", "/api/books/b1/edits/last")).toBeDefined();
    await waitFor(() => expect(screen.getByRole("button", { name: "undo last" })).toBeDisabled());
  });

  it("chapter arrows page the segment browser: › fetches and shows chapter 2", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/chapters/2/segments", {
      chapter_index: 2,
      title: "The Middle",
      segments: [seg({ block_id: "ch002_b0001", text: "Rain fell on the second morning." })],
      edit_warnings: [],
    } satisfies SegmentBrowserOut);
    renderWithProviders(<Review />);

    expect(await screen.findByText("It was a dark night.")).toBeInTheDocument();
    expect(screen.getByText(/chapter 1 of 2 · The Beginning/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "‹" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "›" }));

    expect(await screen.findByText("Rain fell on the second morning.")).toBeInTheDocument();
    expect(screen.getByText(/chapter 2 of 2 · The Middle/)).toBeInTheDocument();
    expect(server.lastCall("GET", "/api/books/b1/chapters/2/segments")).toBeDefined();
    expect(screen.getByRole("button", { name: "›" })).toBeDisabled(); // last chapter
  });

  it("suggest cast previews via POST /assignment/suggest and apply drafts with strategy=smart", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.post("/api/books/b1/assignment/suggest", {
      assignment: ASSIGNMENT,
      would_create_voice_ids: ["v-alice", "v-bob"],
      would_recast_voice_ids: [],
      edit_warnings: [],
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "suggest cast" }));
    expect(await screen.findByText(/2 new/)).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/suggest")).toEqual({ strategy: "smart" });

    // applying writes the draft; the assignment refetch now returns a casting
    server.get("/api/books/b1/assignment", ASSIGNMENT);
    server.post("/api/books/b1/assignment/draft", {
      assignment: ASSIGNMENT,
      created_voice_ids: ["v-alice", "v-bob"],
      edit_warnings: [],
    });
    await user.click(screen.getByRole("button", { name: "apply cast" }));

    expect(await screen.findByText(/created 2 voice\(s\)/)).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
      strategy: "smart",
      recast: false,
      use_llm: false,
    });
  });

  it("applying with AI trait hints sends use_llm=true and a 402 surfaces the payment gate instead of applying", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.post("/api/books/b1/assignment/suggest", {
      assignment: ASSIGNMENT,
      would_create_voice_ids: ["v-alice", "v-bob"],
      would_recast_voice_ids: [],
      edit_warnings: [],
    });
    server.error(
      "POST",
      "/api/books/b1/assignment/draft",
      402,
      "payment_confirmation_required",
      "anthropic cast is a paid call — confirm to proceed",
    );
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "suggest cast" }));
    await user.click(await screen.findByRole("checkbox", { name: /use AI trait hints/ }));
    await user.click(screen.getByRole("button", { name: "apply cast" }));

    expect(await screen.findByText("payment_confirmation_required")).toBeInTheDocument();
    expect(screen.getByText(/anthropic cast is a paid call/)).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
      strategy: "smart",
      recast: false,
      use_llm: true,
    });
    // nothing applied: the preview stays up, still offering apply
    expect(screen.getByRole("button", { name: "apply cast" })).toBeInTheDocument();
    // the paid gate is a hard stop: exactly one draft POST — nothing auto-retries a paid call
    expect(
      server.calls.filter((c) => c.method === "POST" && c.url.includes("/assignment/draft")),
    ).toHaveLength(1);
  });
});
