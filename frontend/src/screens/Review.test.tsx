import { QueryClient } from "@tanstack/react-query";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

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
import { usePlayer } from "../app/usePlayer";
import { makeBook, makeJob } from "../test/fixtures";
import type { MockApi } from "../test/utils";
import { errorResponse, jsonResponse, mockApi, renderWithProviders } from "../test/utils";
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
    unattributed_quote_segments: 0,
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
    unattributed_quote: false,
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

/** Spoiler-safe is off by default; tests about masking switch it on through the View menu. */
async function turnOnSpoilerSafe(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /View/ }));
  await user.click(await screen.findByRole("menuitemcheckbox", { name: /Spoiler-safe/ }));
  // the menu may stay open for more choices; Escape closes it either way
  await user.keyboard("{Escape}");
}

/** The open reassign popover (a plain span, so found through its heading). */
function reassignPopover(): HTMLElement {
  const el = screen.getByText("who says this line?").closest(".popover");
  if (!(el instanceof HTMLElement)) throw new Error("no reassign popover open");
  return el;
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
    expect(screen.getByText(/3 to check in the book · 1 in this chapter/)).toBeInTheDocument();
    expect(screen.getByText(/speakers found by qwen2.5:7b/)).toBeInTheDocument();
    // the chapter page renders segments with the speaker chip in the margin
    expect(screen.getByText("It was a dark night.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ALICE" })).toBeInTheDocument();
  });

  it("an unattributed book explains itself with a Find-the-speakers step and fetches no review data", async () => {
    const server = mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Raw Ingest" })] })
      .get("/api/books/b1", detailFor("b1", "Raw Ingest", false))
      .get("/api/voices", { voices: [], unreadable: [] });
    renderWithProviders(<Review />, { route: "/review?book=b1" });

    expect(await screen.findByText(/hasn't worked out who says each line in this book/)).toBeInTheDocument();
    // the step is done right here: the find-the-speakers panel, not a link away
    const panel = screen.getByRole("region", { name: "Full cast: find the speakers" });
    expect(within(panel).getByRole("button", { name: "▶ find the speakers in the whole book" })).toBeInTheDocument();
    expect(screen.queryByText("stage_prerequisite")).not.toBeInTheDocument();
    expect(server.lastCall("GET", "/characters")).toBeUndefined();
    expect(server.lastCall("GET", "/segments")).toBeUndefined();
  });

  it("a book recorded in one voice but never sent through speaker-finding still gets Find the speakers", async () => {
    const detail = detailFor("b1", "One Voice", false);
    detail.status = { ...detail.status, rendered: true, assembled: true, mastered: true };
    mockApi()
      .get("/api/books", {
        books: [makeBook({ book_id: "b1", title: "One Voice", rendered: true, assembled: true, mastered: true })],
      })
      .get("/api/books/b1", detail)
      .get("/api/voices", { voices: [], unreadable: [] });
    renderWithProviders(<Review />, { route: "/review?book=b1" });

    expect(await screen.findByText(/hasn't worked out who says each line in this book/)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Full cast: find the speakers" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Listen" })).not.toBeInTheDocument();
  });

  it("the no-attributed-book empty state offers Find the speakers even for a book recorded in one voice", async () => {
    mockApi().get("/api/books", {
      books: [makeBook({ book_id: "b3", title: "Solo", rendered: true })],
    });
    renderWithProviders(<Review />);
    expect(await screen.findByRole("region", { name: "Full cast: find the speakers" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Build the .m4b" })).not.toBeInTheDocument();
  });

  it("with no attributed book at all, the empty state points at the first book that needs speakers", async () => {
    const server = mockApi()
      .get("/api/books", {
        books: [
          makeBook({ book_id: "b0", title: "Gone", ingested: false }),
          makeBook({ book_id: "b2", title: "Waiting" }),
        ],
      })
      .post("/api/books/b2/attribute", { job_id: "j-a", kind: "attribute", state: "queued" }, 202);
    renderWithProviders(<Review />);
    const panel = await screen.findByRole("region", { name: "Full cast: find the speakers" });
    expect(screen.getByText("Waiting")).toBeInTheDocument(); // the book it will read, by title
    await userEvent.setup().click(within(panel).getByRole("button", { name: "▶ find the speakers in the whole book" }));
    // the panel starts THAT book, not the shelf's first one
    await waitFor(() => expect(server.lastCall("POST", "/attribute")?.url).toBe("/api/books/b2/attribute"));
  });

  it("with no books at all, the empty state's next step is adding one", async () => {
    mockApi().get("/api/books", { books: [] });
    renderWithProviders(<Review />);
    expect(await screen.findByRole("link", { name: "Add a book" })).toHaveAttribute("href", "/#add");
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
    await turnOnSpoilerSafe(user);
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
    await user.click(within(reassignPopover()).getByRole("button", { name: /Alice/ }));
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

  it("History lists the fixes and undoes the newest after saying exactly what it removes", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    const renameOp = { op: "rename", character_id: "alice", new_name: "Alicia", expected_name: "Alice" };
    let undone = false;
    server.on("GET", "/api/books/b1/edits", () => ({ version: 1, ops: undone ? [] : [renameOp] }));
    server.on("DELETE", "/api/books/b1/edits/last", () => {
      undone = true;
      return { removed: renameOp };
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "History · 1" }));
    const drawer = await screen.findByRole("dialog", { name: "History" });
    expect(within(drawer).getByText('Renamed "Alice" to "Alicia"')).toBeInTheDocument();

    await user.click(within(drawer).getByRole("button", { name: "Undo this…" }));
    // asks first, naming exactly the change it removes; nothing is deleted yet
    const confirm = within(drawer).getByRole("group", { name: "confirm undo" });
    expect(confirm).toHaveTextContent('Undo removes this change: Renamed "Alice" to "Alicia".');
    expect(server.lastCall("DELETE", "/api/books/b1/edits/last")).toBeUndefined();

    await user.click(within(confirm).getByRole("button", { name: "Undo it" }));
    await waitFor(() => expect(server.lastCall("DELETE", "/api/books/b1/edits/last")).toBeDefined());
    expect(await within(drawer).findByText('Undone: Renamed "Alice" to "Alicia"')).toBeInTheDocument();
    expect(await within(drawer).findByText("No fixes yet.")).toBeInTheDocument();
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
    expect(screen.getByRole("button", { name: /Ch 1 · The Beginning/ })).toBeInTheDocument();
    expect(screen.getByText("of 2")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "‹" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "›" }));

    expect(await screen.findByText("Rain fell on the second morning.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Ch 2 · The Middle/ })).toBeInTheDocument();
    expect(server.lastCall("GET", "/api/books/b1/chapters/2/segments")).toBeDefined();
    expect(screen.getByRole("button", { name: "›" })).toBeDisabled(); // last chapter
  });

  it("unattributed quotes join the review queue and render an 'unattributed' chip, not 'narration'", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    const base = chapterOneSegments();
    server.get("/api/books/b1/chapters/1/segments", {
      ...base,
      segments: [
        ...base.segments,
        seg({
          block_id: "ch001_b0002",
          segment_index: 0,
          type: "narration",
          text: '"Who goes there?"',
          confidence: 0,
          unattributed_quote: true,
        }),
      ],
    } satisfies SegmentBrowserOut);
    server.get("/api/books/b1/characters", {
      ...overviewFor("b1", [alice(), bob()]),
      unattributed_quote_segments: 1,
    });
    renderWithProviders(<Review />);

    // distinguishable chip: "speaker unknown", never passed off as plain narration
    const chip = await screen.findByRole("button", { name: "speaker unknown" });
    // the drainstrip counts it: the book-level stat AND the chapter queue (0.0 + the 0.55 dialogue)
    expect(screen.getByText(/1 quote\(s\) with no speaker set · 2 in this chapter/)).toBeInTheDocument();
    // it wears the in-review marker like any low-confidence attribution
    expect(screen.getAllByText(/in review/)).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Next to check ▸" })).toBeEnabled();

    // its chip opens the reassign popover — the queue leads straight to the fix
    await user.click(chip);
    expect(await screen.findByText("who says this line?")).toHaveAttribute("title", "ch001_b0002 [0]");
  });

  it("first cast: one Suggest a cast button (no hash auto-cast) previews, and apply drafts with strategy=smart", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.post("/api/books/b1/assignment/suggest", {
      assignment: ASSIGNMENT,
      would_create_voice_ids: ["v-alice", "v-bob"],
      would_recast_voice_ids: [],
      edit_warnings: [],
    });
    renderWithProviders(<Review />);

    const suggest = await screen.findByRole("button", { name: "Suggest a cast" });
    // the collision-blind hash caster is gone from the UI (it stays in the CLI)
    expect(screen.queryByRole("button", { name: "auto-cast" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "re-draft" })).not.toBeInTheDocument();
    await user.click(suggest);
    expect(await screen.findByText(/2 new/)).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/suggest")).toEqual({
      strategy: "smart",
      stage: "draft",
    });
    // the preview is a table of characters with their proposed voices by name
    const table = screen.getByRole("table", { name: "proposed voices" });
    expect(within(table).getByRole("button", { name: /voice for Alice/ })).toHaveTextContent("auto:Alice");

    // applying writes the draft; the assignment refetch now returns a casting
    server.get("/api/books/b1/assignment", ASSIGNMENT);
    server.post("/api/books/b1/assignment/draft", {
      assignment: ASSIGNMENT,
      created_voice_ids: ["v-alice", "v-bob"],
      edit_warnings: [],
    });
    await user.click(screen.getByRole("button", { name: "apply cast" }));

    expect(await screen.findByText(/made 2 new voices for this cast/)).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
      strategy: "smart",
      stage: "draft",
      accent: "a",
      overrides: {},
      recast: false,
      use_llm: false,
    });
    // applied: the preview closes
    expect(screen.queryByRole("table", { name: "proposed voices" })).not.toBeInTheDocument();
  });

  it("changing a row in the preview re-requests it with an override, and apply sends that override", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.post("/api/books/b1/assignment/suggest", {
      assignment: { ...ASSIGNMENT, assignments: { alice: "alice_auto", bob: "bob_auto" } },
      would_create_voice_ids: ["alice_auto", "bob_auto"],
      would_recast_voice_ids: [],
      edit_warnings: [],
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    const table = await screen.findByRole("table", { name: "proposed voices" });
    expect(within(table).getByRole("button", { name: /voice for Bob/ })).toHaveTextContent("new voice for Bob");

    // pick Bob's voice by hand: the server recomputes the preview with it as an override
    server.post("/api/books/b1/assignment/suggest", {
      assignment: { ...ASSIGNMENT, assignments: { alice: "alice_auto", bob: "v-narr" } },
      would_create_voice_ids: ["alice_auto"],
      would_recast_voice_ids: [],
      edit_warnings: [],
    });
    await user.click(within(table).getByRole("button", { name: /voice for Bob/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Narrator · kokoro" }));
    await waitFor(() =>
      expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/suggest")).toEqual({
        strategy: "smart",
        stage: "draft",
        accent: "a",
        overrides: { bob: "v-narr" },
      }),
    );
    const bobRow = await screen.findByRole("row", { name: /Bob.*your pick/ });
    expect(within(bobRow).getByRole("button", { name: /voice for Bob/ })).toHaveTextContent("Narrator");

    server.get("/api/books/b1/assignment", ASSIGNMENT);
    server.post("/api/books/b1/assignment/draft", {
      assignment: ASSIGNMENT,
      created_voice_ids: ["alice_auto"],
      edit_warnings: [],
    });
    await user.click(screen.getByRole("button", { name: "apply cast" }));
    await waitFor(() => expect(draftPosts(server)).toHaveLength(1));
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
      strategy: "smart",
      stage: "draft",
      accent: "a",
      overrides: { bob: "v-narr" },
      recast: false,
      use_llm: false,
    });
  });

  it("a failed preview is shown in plain words and nothing is retried", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.error("POST", "/api/books/b1/assignment/suggest", 422, "invalid", "not enough distinct voices for this cast");
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    expect(await screen.findByText("couldn't suggest a cast")).toBeInTheDocument();
    expect(screen.getByText(/not enough distinct voices/)).toBeInTheDocument();
    expect(server.calls.filter((c) => c.url.includes("/assignment/suggest"))).toHaveLength(1);
  });

  it("an older preview reply that lands after a newer one never replaces the table", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    let releaseSlow: () => void = () => {};
    const slow = new Promise<void>((resolve) => (releaseSlow = resolve));
    const reply = (assignments: Record<string, string>) =>
      jsonResponse({
        assignment: { ...ASSIGNMENT, assignments },
        would_create_voice_ids: [],
        would_recast_voice_ids: [],
        edit_warnings: [],
      });
    server.on("POST", "/api/books/b1/assignment/suggest", async (_url, init) => {
      const overrides = (JSON.parse(String(init?.body)).overrides ?? {}) as Record<string, string>;
      if (overrides.alice) return reply({ alice: overrides.alice, bob: overrides.bob ?? "v-bob" });
      if (overrides.bob) {
        await slow; // the Bob change is answered late — after the Alice change that followed it
        return reply({ alice: "v-alice", bob: overrides.bob });
      }
      return reply({ alice: "v-alice", bob: "v-bob" });
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    const table = await screen.findByRole("table", { name: "proposed voices" });
    await user.click(within(table).getByRole("button", { name: /voice for Bob/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Narrator · kokoro" }));
    await user.click(within(table).getByRole("button", { name: /voice for Alice/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "auto:Bob · kokoro" }));
    await waitFor(() =>
      expect(within(table).getByRole("button", { name: /voice for Alice/ })).toHaveTextContent("auto:Bob"),
    );

    releaseSlow();
    await waitFor(() => expect(screen.getByRole("status")).toBeEmptyDOMElement());
    // the stale Bob-only reply (Alice back on her own voice) must not win
    expect(within(table).getByRole("button", { name: /voice for Alice/ })).toHaveTextContent("auto:Bob");
    expect(within(table).getByRole("button", { name: /voice for Bob/ })).toHaveTextContent("Narrator");
  });

  it("switching books while a suggestion is on its way drops it — no preview opens on the other book", async () => {
    const user = userEvent.setup();
    const server = mockApi().get("/api/books", {
      books: [
        makeBook({ book_id: "b1", title: "Book One", attributed: true }),
        makeBook({ book_id: "b2", title: "Book Two", attributed: true }),
      ],
    });
    registerBook(server, "b1", "Book One");
    registerBook(server, "b2", "Book Two");
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    server.on("POST", "/api/books/b1/assignment/suggest", async () => {
      await gate;
      return jsonResponse({
        assignment: ASSIGNMENT,
        would_create_voice_ids: ["v-alice", "v-bob"],
        would_recast_voice_ids: [],
        edit_warnings: [],
      });
    });
    renderWithProviders(<Review />, { route: "/review?book=b1" });

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    await waitFor(() => expect(server.lastCall("POST", "/api/books/b1/assignment/suggest")).toBeDefined());
    await user.click(screen.getByRole("button", { name: /Book One/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Book Two" }));
    await waitFor(() => expect(server.lastCall("GET", "/api/books/b2/characters")).toBeDefined());

    release();
    // let the late reply settle, then check it opened nothing on Book Two
    await waitFor(() => expect(screen.getByRole("button", { name: "Suggest a cast" })).toBeEnabled());
    await gate;
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByRole("table", { name: "proposed voices" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "suggested cast" })).not.toBeInTheDocument();
  });

  it("a refused row change is undone in plain words, not retried, and apply sends the last good choices", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.post("/api/books/b1/assignment/suggest", {
      assignment: ASSIGNMENT,
      would_create_voice_ids: ["v-alice", "v-bob"],
      would_recast_voice_ids: [],
      edit_warnings: [],
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    const table = await screen.findByRole("table", { name: "proposed voices" });
    server.error("POST", "/api/books/b1/assignment/suggest", 422, "invalid", "that voice can't be used here");
    await user.click(within(table).getByRole("button", { name: /voice for Bob/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Narrator · kokoro" }));

    const status = screen.getByRole("status");
    await waitFor(() => expect(status).toHaveTextContent(/couldn't use that change, so it was undone/));
    expect(status).toHaveTextContent(/that voice can't be used here/);
    expect(status).not.toHaveTextContent("invalid");
    // the panel reports it; the first-request failure box stays away
    expect(screen.queryByText("couldn't suggest a cast")).not.toBeInTheDocument();
    expect(server.calls.filter((c) => c.url.includes("/assignment/suggest"))).toHaveLength(2);

    server.get("/api/books/b1/assignment", ASSIGNMENT);
    server.post("/api/books/b1/assignment/draft", {
      assignment: ASSIGNMENT,
      created_voice_ids: [],
      edit_warnings: [],
    });
    await user.click(screen.getByRole("button", { name: "apply cast" }));
    await waitFor(() => expect(draftPosts(server)).toHaveLength(1));
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toMatchObject({ overrides: {} });
  });

  it("Suggest a cast can't throw away an open preview's changes", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.post("/api/books/b1/assignment/suggest", {
      assignment: ASSIGNMENT,
      would_create_voice_ids: ["v-alice", "v-bob"],
      would_recast_voice_ids: [],
      edit_warnings: [],
    });
    renderWithProviders(<Review />);

    const suggest = await screen.findByRole("button", { name: "Suggest a cast" });
    await user.click(suggest);
    await screen.findByRole("table", { name: "proposed voices" });
    expect(suggest).toBeDisabled();
    // dismissing the preview frees it again
    await user.click(screen.getByRole("button", { name: "dismiss" }));
    expect(suggest).toBeEnabled();
  });

  it("applying with AI trait hints sends use_llm=true; a 402 gates on explicit approval, then retries with confirm_paid", async () => {
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

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    await user.click(await screen.findByRole("checkbox", { name: /use AI trait hints/ }));
    await user.click(screen.getByRole("button", { name: "apply cast" }));

    // plain words, not the server code
    expect(await screen.findByText("the AI caster is paid — approve it below to continue")).toBeInTheDocument();
    expect(screen.queryByText("payment_confirmation_required")).not.toBeInTheDocument();
    expect(screen.queryByText(/anthropic cast is a paid call/)).not.toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
      strategy: "smart",
      stage: "draft",
      accent: "a",
      overrides: {},
      recast: false,
      use_llm: true,
    });
    // nothing applied: the preview stays up, still offering apply
    expect(screen.getByRole("button", { name: "apply cast" })).toBeInTheDocument();
    // the paid gate is a hard stop: exactly one draft POST — nothing auto-retries a paid call
    expect(
      server.calls.filter((c) => c.method === "POST" && c.url.includes("/assignment/draft")),
    ).toHaveLength(1);

    // the recourse: retry is dead until the paid caster is explicitly approved
    const retry = screen.getByRole("button", { name: "retry cast" });
    expect(retry).toBeDisabled();
    server.post("/api/books/b1/assignment/draft", {
      assignment: ASSIGNMENT,
      created_voice_ids: ["v-alice", "v-bob"],
      edit_warnings: [],
    });
    await user.click(screen.getByRole("checkbox", { name: /approve the paid \(Anthropic\) caster/ }));
    await user.click(retry);

    expect(await screen.findByText(/made 2 new voices for this cast/)).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
      strategy: "smart",
      stage: "draft",
      accent: "a",
      overrides: {},
      recast: false,
      use_llm: true,
      confirm_paid: true,
    });
    // success dismisses the preview panel — the paid flow ends cleanly
    expect(screen.queryByRole("button", { name: "apply cast" })).not.toBeInTheDocument();
  });

  it("tag chips narrow every voice picker (AND semantics) but never hide the assigned voice", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/assignment", ASSIGNMENT);
    server.get("/api/voices", {
      voices: [
        { ...voice("v-narr", "Narrator"), tags: ["male", "deep"] },
        { ...voice("v-alice", "auto:Alice"), tags: ["female", "young"] },
        { ...voice("v-lily", "Lily"), tags: ["female"] },
        { ...voice("v-bob", "auto:Bob"), tags: ["male"] },
      ],
      unreadable: [],
    });
    renderWithProviders(<Review />);

    // chips come from the union of tags; picking one narrows the pool
    await user.click(await screen.findByRole("button", { name: "female" }));
    expect(screen.getByText("2 of 4 voices")).toBeInTheDocument();

    // Bob's picker: the two female-tagged voices plus his own (kept despite the filter) — not the narrator
    const bobRow = await screen.findByRole("row", { name: /Bob/ });
    await user.click(within(bobRow).getByRole("button", { name: /auto:Bob · kokoro/ }));
    const listbox = await screen.findByRole("listbox");
    expect(within(listbox).getByRole("option", { name: "auto:Bob · kokoro" })).toBeInTheDocument();
    expect(within(listbox).getByRole("option", { name: "auto:Alice · kokoro" })).toBeInTheDocument();
    expect(within(listbox).getByRole("option", { name: "Lily · kokoro" })).toBeInTheDocument();
    expect(within(listbox).queryByRole("option", { name: "Narrator · kokoro" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");

    // a second tag ANDs: only voices carrying BOTH remain
    await user.click(screen.getByRole("button", { name: "young" }));
    expect(screen.getByText("1 of 4 voices")).toBeInTheDocument();

    // clear restores the full pool
    await user.click(screen.getByRole("button", { name: "clear" }));
    expect(screen.queryByText(/of 4 voices/)).not.toBeInTheDocument();
  });

  it("saving a cast change on a rendered book prompts a re-render with a deep link", async () => {
    const user = userEvent.setup();
    const base = detailFor("b1", "Attributed One");
    const rendered: BookDetail = {
      ...base,
      status: { ...base.status, assigned: true, rendered: true },
    };
    let current: VoiceAssignment = ASSIGNMENT;
    const server = mockApi()
      .get("/api/books", {
        books: [makeBook({ book_id: "b1", title: "Attributed One", attributed: true, assigned: true, rendered: true })],
      })
      .get("/api/books/b1", rendered)
      .get("/api/books/b1/characters", overviewFor("b1", [alice(), bob()]))
      .get("/api/books/b1/chapters/1/segments", chapterOneSegments())
      .get("/api/books/b1/attribution", chapterOneAttribution())
      .get("/api/books/b1/edits", { version: 1, ops: [] })
      .get("/api/voices", {
        voices: [voice("v-narr", "Narrator"), voice("v-alice", "auto:Alice"), voice("v-bob", "auto:Bob")],
        unreadable: [],
      });
    server.on("GET", "/api/books/b1/assignment", () => jsonResponse(current));
    server.on("PUT", "/api/books/b1/assignment", (_u, init) => {
      current = { ...current, ...(JSON.parse(String(init?.body)) as Partial<VoiceAssignment>) };
      return jsonResponse(current);
    });
    renderWithProviders(<Review />);

    // reassign Alice's voice (auto:Alice -> Narrator): this dirties the local casting
    const aliceRow = await screen.findByRole("row", { name: /Alice/ });
    await user.click(within(aliceRow).getByRole("button", { name: /auto:Alice · kokoro/ }));
    const listbox = await screen.findByRole("listbox");
    await user.click(within(listbox).getByRole("option", { name: "Narrator · kokoro" }));

    await user.click(screen.getByRole("button", { name: "save casting" }));

    // the rendered audio is now stale — the screen says so and deep-links to Render & Jobs
    expect(
      await screen.findByText(/the rendered audio still uses the old ones/),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /re-record on Produce/ })).toHaveAttribute(
      "href",
      "/render?book=b1",
    );
  });
});

/* ------------------------------------------------------------------ cast reset (keep_existing) */

/** Alice stays on her auto voice; Bob is a hand pick (Lily); the narrator is hand-chosen. */
const MIXED: VoiceAssignment = {
  ...ASSIGNMENT,
  narrator_voice_id: "v-narr",
  assignments: { alice: "alice_auto", bob: "v-lily" },
};

function setupCast(assignment: VoiceAssignment): MockApi {
  const server = setupReview();
  server.get("/api/books/b1/assignment", assignment).get("/api/voices", {
    voices: [
      voice("v-narr", "Narrator"),
      voice("alice_auto", "auto:Alice"),
      voice("v-lily", "Lily"),
      voice("narrator_af_heart", "Default Narrator"),
      voice("narrator_ab12", "Narrator"), // a USER voice named "Narrator": slugify(name)_<hex>
    ],
    unreadable: [],
  });
  // the reset dialog predicts the narrator a reset writes: narrator_{slugify(default preset)}
  server.get("/api/settings", { kokoro_default_voice: "af_heart" });
  server.post("/api/books/b1/assignment/draft", {
    assignment: { ...assignment, assignments: { alice: "alice_auto", bob: "bob_auto" } },
    created_voice_ids: [],
    edit_warnings: [],
  });
  return server;
}

const draftPosts = (server: MockApi) =>
  server.calls.filter((c) => c.method === "POST" && c.url.includes("/assignment/draft"));

describe("Review cast reset", () => {
  it("re-draft is folded into Suggest a cast; the stage chips live under Advanced", async () => {
    const user = userEvent.setup();
    const server = setupCast(MIXED);
    renderWithProviders(<Review />);

    // the saved cast has loaded once reset is offered
    await screen.findByRole("button", { name: "reset cast…" });
    expect(screen.getByRole("button", { name: "Suggest a cast" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "re-draft" })).not.toBeInTheDocument();
    // draft/final are kept (they matter for paid final casts) but tucked into a disclosure
    const advanced = screen.getByText(/advanced · cast stage: draft/);
    expect(screen.getByRole("button", { name: "final" })).not.toBeVisible();
    await user.click(advanced);
    await user.click(screen.getByRole("button", { name: "final" }));
    expect(screen.getByRole("button", { name: "save casting" })).toBeEnabled();
    expect(draftPosts(server)).toHaveLength(0);
  });

  it("a final cast stays final through the preview and the apply", async () => {
    const user = userEvent.setup();
    const server = setupCast({ ...MIXED, stage: "final" });
    server.post("/api/books/b1/assignment/suggest", {
      assignment: { ...MIXED, stage: "final" },
      would_create_voice_ids: [],
      would_recast_voice_ids: ["alice_auto"],
      edit_warnings: [],
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    await screen.findByRole("table", { name: "proposed voices" });
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/suggest")).toMatchObject({ stage: "final" });
    // nothing differs from the saved cast until re-cast is ticked (with its re-recording warning)
    expect(screen.getByRole("button", { name: "nothing to apply" })).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: /re-cast the 1 existing \(their lines are recorded again\)/ }));
    await user.click(screen.getByRole("button", { name: "apply cast" }));
    await waitFor(() => expect(draftPosts(server)).toHaveLength(1));
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toMatchObject({
      strategy: "smart",
      stage: "final",
      recast: true,
    });
  });

  it("reset cast… lists the manual picks and a hand-chosen narrator; cancel posts nothing", async () => {
    const user = userEvent.setup();
    const server = setupCast(MIXED);
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "reset cast…" }));
    const dialog = await screen.findByRole("dialog", { name: "Reset cast" });
    const losses = await within(dialog).findByRole("list", { name: "what a reset discards" });
    // only Bob is a hand pick — Alice is on alice_auto and is not listed
    expect(within(losses).getByText("1 manual pick: Bob → Lily")).toBeInTheDocument();
    expect(within(losses).getByText("narrator: Narrator")).toBeInTheDocument();
    expect(within(losses).queryByText(/Alice/)).not.toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(draftPosts(server)).toHaveLength(0);
  });

  it("confirming the reset posts keep_existing:false with the smart caster and closes the dialog", async () => {
    const user = userEvent.setup();
    const server = setupCast(MIXED);
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "reset cast…" }));
    const dialog = await screen.findByRole("dialog", { name: "Reset cast" });
    const confirm = within(dialog).getByRole("button", { name: "reset cast" });
    await waitFor(() => expect(confirm).toBeEnabled()); // waits for the settings check
    await user.click(confirm);

    await waitFor(() => expect(draftPosts(server)).toHaveLength(1));
    // never the collision-blind hash caster (the server default) — it is CLI-only now
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
      strategy: "smart",
      stage: "draft",
      keep_existing: false,
    });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("an all-auto casting says there is nothing manual to lose", async () => {
    const user = userEvent.setup();
    setupCast({
      ...ASSIGNMENT,
      narrator_voice_id: "narrator_af_heart",
      assignments: { alice: "alice_auto" },
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "reset cast…" }));
    const dialog = await screen.findByRole("dialog", { name: "Reset cast" });
    expect(await within(dialog).findByText(/nothing manual to lose/)).toBeInTheDocument();
    expect(within(dialog).queryByRole("list")).not.toBeInTheDocument();
  });

  it("a user voice named Narrator (narrator_<hex>) is listed as lost, not mistaken for auto", async () => {
    const user = userEvent.setup();
    setupCast({ ...ASSIGNMENT, narrator_voice_id: "narrator_ab12", assignments: { alice: "alice_auto" } });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "reset cast…" }));
    const dialog = await screen.findByRole("dialog", { name: "Reset cast" });
    const losses = await within(dialog).findByRole("list", { name: "what a reset discards" });
    expect(within(losses).getByText("narrator: Narrator")).toBeInTheDocument();
    expect(within(dialog).queryByText(/nothing manual to lose/)).not.toBeInTheDocument();
  });

  it("suggest cast tells you your hand picks survive the apply", async () => {
    const user = userEvent.setup();
    const server = setupCast(MIXED);
    server.post("/api/books/b1/assignment/suggest", {
      assignment: MIXED,
      would_create_voice_ids: [],
      would_recast_voice_ids: ["alice_auto"],
      edit_warnings: [],
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    expect(await screen.findByText("Your 1 pick is kept")).toBeInTheDocument();
    // the existing create/recast summary is unchanged
    // "already cast" counts the saved cast (Alice and Bob), not the voices that exist
    expect(screen.getByText(/0 new · 2 already cast/)).toBeInTheDocument();
  });
});

/* ------------------------------------------------------------------ hear a voice where you cast it */

/** A cast book: Alice on v-alice, Bob on v-bob, narrated by v-narr (all local kokoro voices). */
function setupHear(): MockApi {
  const server = setupReview();
  server.get("/api/books/b1/assignment", ASSIGNMENT);
  return server;
}

const auditionPosts = (server: MockApi, voiceId?: string) =>
  server.calls.filter(
    (c) =>
      c.method === "POST" &&
      c.url.endsWith("/audition") &&
      (!voiceId || c.url === `/api/voices/${voiceId}/audition`),
  );
const bodyOf = (call: { body: unknown }) => JSON.parse(call.body as string) as Record<string, unknown>;
const tick = () => new Promise((r) => setTimeout(r, 60));
/** The visible refusal box's sentence (the row's screen-reader live region repeats it). */
const shown = (re: RegExp) => screen.findByText(re, { selector: ".refusal *" });
/** What the row's always-mounted live region currently announces. */
const announced = (row: HTMLElement) => row.querySelector("[aria-live]")?.textContent ?? "";

describe("Review: hear a voice where you cast it", () => {
  afterEach(() => vi.restoreAllMocks());

  it("▶ on a roster row plays the cast voice speaking that character's first line", async () => {
    const user = userEvent.setup();
    const play = vi.spyOn(HTMLMediaElement.prototype, "play");
    const server = setupHear().post("/api/voices/v-alice/audition", {
      voice_id: "v-alice",
      duration_seconds: 1.2,
      cost_usd: 0,
      audition_url: "/api/voices/v-alice/audition.wav",
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "hear Alice's voice" }));
    await waitFor(() => expect(play).toHaveBeenCalled());
    const el = play.mock.contexts.at(-1) as HTMLAudioElement;
    expect(el.src).toContain("/api/voices/v-alice/audition.wav?t=");
    const posts = auditionPosts(server);
    expect(posts).toHaveLength(1);
    expect(posts[0].url).toBe("/api/voices/v-alice/audition");
    // a fresh click is never a paid confirmation; the line is Alice's own
    expect(bodyOf(posts[0])).toEqual({ confirm_paid: false, text: '"Hello there," she said.' });
  });

  it("a character with no sample line plays the default sentence; an unsaved pick is what plays", async () => {
    const user = userEvent.setup();
    const server = setupHear().post(/\/api\/voices\/[^/]+\/audition$/, {
      voice_id: "v-narr",
      duration_seconds: 1,
      cost_usd: 0,
      audition_url: "/x",
    });
    renderWithProviders(<Review />);

    // re-cast Bob to the Narrator voice WITHOUT saving: the ▶ follows the picker
    const bobRow = await screen.findByRole("row", { name: /Bob/ });
    await user.click(within(bobRow).getByRole("button", { name: /auto:Bob · kokoro/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Narrator · kokoro" }));
    await user.click(screen.getByRole("button", { name: "hear Bob's voice" }));

    await waitFor(() => expect(auditionPosts(server)).toHaveLength(1));
    expect(auditionPosts(server)[0].url).toBe("/api/voices/v-narr/audition");
    expect(bodyOf(auditionPosts(server)[0])).toEqual({ confirm_paid: false }); // no text → server default
    expect(server.calls.some((c) => c.method === "PUT")).toBe(false);
  });

  it("engine_cold offers Download now, which queues the voice engine's warm-up job", async () => {
    const user = userEvent.setup();
    const server = setupHear()
      .error("POST", "/api/voices/v-alice/audition", 409, "engine_cold", "kokoro has never loaded")
      .post(
        "/api/engines/kokoro/warmup",
        makeJob({ job_id: "w1", kind: "warmup", state: "queued", book_id: "engine:kokoro" }),
      );
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "hear Alice's voice" }));
    expect(await shown(/Kokoro hasn't been downloaded yet/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Download now" }));
    await waitFor(() => expect(server.lastCall("POST", "/warmup")?.url).toBe("/api/engines/kokoro/warmup"));
    expect(await shown(/The Kokoro download is queued — it starts on its own/)).toBeInTheDocument();
    // the download is a job; the sample is NOT re-tried behind the user's back
    expect(auditionPosts(server)).toHaveLength(1);
  });

  it("the queued download is followed to its end: ready says so, a failure says why, both dismissable", async () => {
    const user = userEvent.setup();
    const job = makeJob({ job_id: "w1", kind: "warmup", state: "queued", book_id: "engine:kokoro" });
    let current = job;
    const server = setupHear()
      .error("POST", "/api/voices/v-alice/audition", 409, "engine_cold", "kokoro has never loaded")
      .post("/api/engines/kokoro/warmup", job);
    server.on("GET", "/api/jobs/w1", () => jsonResponse(current));
    const { queryClient } = renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "hear Alice's voice" }));
    await user.click(await screen.findByRole("button", { name: "Download now" }));
    expect(await shown(/The Kokoro download is queued/)).toBeInTheDocument();

    current = { ...job, state: "succeeded", is_terminal: true, finished_at: "2026-07-11T00:05:00Z" };
    await queryClient.refetchQueries({ queryKey: ["jobs", "one", "w1"] });
    expect(await shown(/Kokoro is downloaded and ready — press ▶/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "dismiss" }));
    expect(screen.queryByText(/downloaded and ready/, { selector: ".refusal *" })).not.toBeInTheDocument();

    // again, and this time the job fails: the row says so (no stale "downloading") and offers the download again
    await user.click(screen.getByRole("button", { name: "hear Alice's voice" }));
    await user.click(await screen.findByRole("button", { name: "Download now" }));
    current = { ...job, state: "failed", is_terminal: true, error: "ffmpeg not found on PATH" };
    await queryClient.refetchQueries({ queryKey: ["jobs", "one", "w1"] });
    expect(await shown(/Couldn't get Kokoro ready/)).toBeInTheDocument();
    expect(screen.queryByText(/Downloading Kokoro/, { selector: ".refusal *" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "download again" })).toBeInTheDocument();
    expect(auditionPosts(server)).toHaveLength(2); // one per ▶ click, never on its own
  });

  it("pressing ▶ a second time plays the NEW take and shows ■ even when pause events fire late", async () => {
    const user = userEvent.setup();
    // browser-like media: pause() dispatches 'pause' asynchronously and an interrupted play() rejects
    const made: HTMLAudioElement[] = [];
    vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(function (this: HTMLMediaElement) {
      made.push(this as HTMLAudioElement);
      return new Promise<void>((resolve, reject) => {
        (this as unknown as { _reject: (e: Error) => void })._reject = reject;
        setTimeout(resolve, 200);
      });
    });
    vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(function (this: HTMLMediaElement) {
      const el = this as unknown as { _reject?: (e: Error) => void } & HTMLMediaElement;
      setTimeout(() => {
        el._reject?.(new DOMException("interrupted", "AbortError"));
        el.dispatchEvent(new Event("pause"));
      }, 0);
    });
    setupHear().post("/api/voices/v-alice/audition", {
      voice_id: "v-alice",
      duration_seconds: 1,
      cost_usd: 0,
      audition_url: "/x",
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "hear Alice's voice" }));
    const stop = await screen.findByRole("button", { name: "stop Alice's voice" });
    made[0].dispatchEvent(new Event("ended"));
    await waitFor(() => expect(stop).toHaveAccessibleName("hear Alice's voice"));

    await user.click(stop);
    await waitFor(() => expect(made).toHaveLength(2));
    await tick();
    // exactly one new element, for a new take — the first take is never replayed…
    expect(made).toHaveLength(2);
    expect(made[1].src).not.toBe(made[0].src);
    // …and the stale element's late pause/AbortError can't flip the ■ of the take now playing
    expect(screen.getByRole("button", { name: "stop Alice's voice" })).toBe(stop);
  });

  it("the ▶ keeps keyboard focus while the sample is made, and gets it back after a recourse click", async () => {
    const user = userEvent.setup();
    let release: () => void = () => {};
    const server = setupHear();
    server.on(
      "POST",
      "/api/voices/v-alice/audition",
      () =>
        new Promise<Response>((resolve) => {
          release = () => resolve(errorResponse(409, "gpu_busy", "a job holds the GPU"));
        }),
    );
    renderWithProviders(<Review />);

    const key = await screen.findByRole("button", { name: "hear Alice's voice" });
    key.focus();
    await user.keyboard("{Enter}");
    // still focused and only aria-disabled while it works (never `disabled`, which drops focus)
    await waitFor(() => expect(key).toHaveAttribute("aria-busy", "true"));
    expect(key).not.toBeDisabled();
    expect(key).toHaveAttribute("aria-disabled", "true");
    expect(key).toHaveFocus();
    await user.keyboard("{Enter}"); // ignored while working
    release();
    await shown(/Busy recording/);
    expect(auditionPosts(server)).toHaveLength(1);

    // the recourse removes its own button; focus comes home to the ▶, not <body>
    screen.getByRole("button", { name: "dismiss" }).focus();
    await user.keyboard("{Enter}");
    await waitFor(() => expect(key).toHaveFocus());
  });

  it("a paid voice refused as busy quotes its price again: try again is never a confirmed call", async () => {
    const user = userEvent.setup();
    const server = setupHear().get("/api/voices", {
      voices: [
        voice("v-narr", "Narrator"),
        { ...voice("v-alice", "Cloud Alice"), kind: "preset", engine: "elevenlabs", preset_id: "stock1" },
        voice("v-bob", "auto:Bob"),
      ],
      unreadable: [],
    });
    let busy = true;
    server.on("POST", "/api/voices/v-alice/audition", (_u, init) => {
      const paid = (JSON.parse(String(init?.body)) as { confirm_paid: boolean }).confirm_paid;
      if (!paid) return errorResponse(402, "payment_confirmation_required", "a paid call", { estimated_usd: 0.0123 });
      return busy
        ? errorResponse(409, "cloud_busy", "voice slots in use")
        : jsonResponse({ voice_id: "v-alice", duration_seconds: 2, cost_usd: 0.0123, audition_url: "/x" });
    });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "hear Alice's voice" }));
    await user.click(await screen.findByRole("button", { name: /confirm ~\$0\.0123 & play/ }));
    expect(await shown(/A recording is using the cloud voices/)).toBeInTheDocument();

    busy = false;
    await user.click(screen.getByRole("button", { name: "try again" }));
    // the retry is unconfirmed, so the price is named again and needs its own click
    expect(await shown(/billed per use — this sample costs about \$0\.0123/)).toBeInTheDocument();
    const bodies = auditionPosts(server).map(bodyOf);
    expect(bodies.map((b) => b.confirm_paid)).toEqual([false, true, false]);
  });

  it("gpu_busy says what is busy, in plain words, and is never retried until clicked", async () => {
    const user = userEvent.setup();
    const server = setupHear().error(
      "POST",
      "/api/voices/v-alice/audition",
      409,
      "gpu_busy",
      "a render job is running; the single GPU cannot also host an audition",
      makeJob({ kind: "render", state: "running" }),
    );
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "hear Alice's voice" }));
    expect(await shown(/Busy recording — try after it finishes/)).toBeInTheDocument();
    // announced through the row's always-mounted live region
    expect(announced(screen.getByRole("row", { name: /Alice/ }))).toMatch(/Busy recording/);
    expect(screen.queryByText("gpu_busy")).not.toBeInTheDocument(); // the code is a tooltip only
    await tick();
    await tick();
    expect(auditionPosts(server)).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "try again" }));
    await waitFor(() => expect(auditionPosts(server)).toHaveLength(2));
    expect(bodyOf(auditionPosts(server)[1]).confirm_paid).toBe(false);
  });

  it("audition_in_flight asks to wait a moment", async () => {
    const user = userEvent.setup();
    const server = setupHear().error(
      "POST",
      "/api/voices/v-alice/audition",
      409,
      "audition_in_flight",
      "another audition is already synthesizing",
    );
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "hear Alice's voice" }));
    expect(await shown(/Another voice sample is being made — wait a moment/)).toBeInTheDocument();
    await tick();
    expect(auditionPosts(server)).toHaveLength(1);
  });

  it("a paid voice names its price and plays only after the confirm click", async () => {
    const user = userEvent.setup();
    const play = vi.spyOn(HTMLMediaElement.prototype, "play");
    const server = setupHear().get("/api/voices", {
      voices: [
        voice("v-narr", "Narrator"),
        { ...voice("v-alice", "Cloud Alice"), kind: "preset", engine: "elevenlabs", preset_id: "stock1" },
        voice("v-bob", "auto:Bob"),
      ],
      unreadable: [],
    });
    server.on("POST", "/api/voices/v-alice/audition", (_u, init) =>
      (JSON.parse(String(init?.body)) as { confirm_paid: boolean }).confirm_paid
        ? jsonResponse({ voice_id: "v-alice", duration_seconds: 2, cost_usd: 0.0123, audition_url: "/x" })
        : errorResponse(402, "payment_confirmation_required", "a paid call", { estimated_usd: 0.0123 }),
    );
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "hear Alice's voice" }));
    expect(await shown(/billed per use — this sample costs about \$0\.0123/)).toBeInTheDocument();
    await tick();
    // nothing paid happened on its own
    expect(auditionPosts(server)).toHaveLength(1);
    expect(play).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /confirm ~\$0\.0123 & play/ }));
    await waitFor(() => expect(play).toHaveBeenCalled());
    const posts = auditionPosts(server);
    expect(posts).toHaveLength(2);
    expect(bodyOf(posts[1])).toEqual({ confirm_paid: true, text: '"Hello there," she said.' });
  });

  it("▶ in the Suggest-a-cast table plays the proposed voice with the character's line", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server
      .post("/api/books/b1/assignment/suggest", {
        assignment: ASSIGNMENT,
        would_create_voice_ids: [],
        would_recast_voice_ids: [],
        edit_warnings: [],
      })
      .post("/api/voices/v-alice/audition", {
        voice_id: "v-alice",
        duration_seconds: 1,
        cost_usd: 0,
        audition_url: "/x",
      });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Suggest a cast" }));
    const table = await screen.findByRole("table", { name: "proposed voices" });
    await user.click(within(table).getByRole("button", { name: "hear Alice's proposed voice" }));
    await waitFor(() => expect(auditionPosts(server, "v-alice")).toHaveLength(1));
    expect(bodyOf(auditionPosts(server)[0])).toEqual({ confirm_paid: false, text: '"Hello there," she said.' });
  });

  it("+ new voice for Alice opens Add voice over Review, prefilled, and keeps the unsaved casting", async () => {
    const user = userEvent.setup();
    let voices = [voice("v-narr", "Narrator"), voice("v-alice", "auto:Alice"), voice("v-bob", "auto:Bob")];
    const server = setupHear();
    server.on("GET", "/api/voices", () => jsonResponse({ voices, unreadable: [] }));
    server.on("POST", "/api/voices", () => {
      const created = voice("v-new", "Alice");
      voices = [...voices, created];
      return jsonResponse(created);
    });
    server.on("PUT", "/api/books/b1/assignment", (_u, init) =>
      jsonResponse({ ...ASSIGNMENT, ...(JSON.parse(String(init?.body)) as Partial<VoiceAssignment>) }),
    );
    renderWithProviders(<Review />);

    // an unsaved change first: Bob → Narrator
    const bobRow = await screen.findByRole("row", { name: /Bob/ });
    await user.click(within(bobRow).getByRole("button", { name: /auto:Bob · kokoro/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Narrator · kokoro" }));
    expect(screen.getByRole("button", { name: "save casting" })).toBeEnabled();

    // Alice's picker offers a new voice of her own
    const aliceRow = screen.getByRole("row", { name: /Alice/ });
    await user.click(within(aliceRow).getByRole("button", { name: /auto:Alice · kokoro/ }));
    await user.click(
      within(await screen.findByRole("listbox")).getByRole("option", { name: "+ new voice for Alice" }),
    );

    const dialog = await screen.findByRole("dialog", { name: "New voice for Alice" });
    expect(within(dialog).getByPlaceholderText("Narrator")).toHaveValue("Alice");
    await user.click(within(dialog).getByRole("button", { name: "add voice" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    // a recipe from her name, seeded with her gender
    expect(server.jsonBodyOf("POST", "/api/voices")).toEqual({
      kind: "blend",
      name: "Alice",
      gender: "female",
      accent: "a",
    });
    // Bob's unsaved pick survived, and Alice is now on the new voice — still unsaved
    expect(
      within(screen.getByRole("row", { name: /Bob/ })).getByRole("button", { name: /Narrator · kokoro/ }),
    ).toBeInTheDocument();
    expect(
      await within(screen.getByRole("row", { name: /Alice/ })).findByRole("button", { name: /^Alice · kokoro/ }),
    ).toBeInTheDocument();
    expect(server.calls.some((c) => c.method === "PUT")).toBe(false);

    await user.click(screen.getByRole("button", { name: "save casting" }));
    await waitFor(() => expect(server.lastCall("PUT", "/api/books/b1/assignment")).toBeDefined());
    expect(server.jsonBodyOf("PUT", "/api/books/b1/assignment")).toMatchObject({
      assignments: { alice: "v-new", bob: "v-narr" },
    });
  });
});

/* ------------------------------------------------------------------ 1.10 navigation */

/** Renders the router's current query string so tests can read what the screen wrote there. */
function LocationProbe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.search}</output>;
}
const urlParams = () => new URLSearchParams(screen.getByTestId("loc").textContent ?? "");

/** The shared player's playlist, read out ("clips|title|speaker"); `preload` loads a chapter
    first, as Listen would, paused. */
function PlayerProbe({
  preload,
}: {
  preload?: { title: string; clips: { src: string; duration: number; key: string; speaker: string; words: [] }[] };
}) {
  const p = usePlayer();
  if (!p) return null;
  const clip = p.clips[p.index];
  return (
    <>
      <output data-testid="player">{`${p.clips.length}|${p.chapterTitle}|${clip?.speaker ?? ""}`}</output>
      {preload && (
        <button onClick={() => p.load("b1", preload.title, preload.clips, { autoplay: false })}>preload</button>
      )}
    </>
  );
}

function mallory(): CharacterSummary {
  return { ...bob(), id: "mallory", name: "Mallory", sample_lines: [], first_appearance: "ch005_b0010" };
}

/** Chapter 2's full page: narration, then a low-confidence Bob line. */
function chapterTwoSegments(): SegmentBrowserOut {
  return {
    chapter_index: 2,
    title: "The Middle",
    segments: [
      seg({ block_id: "ch002_b0001", text: "Rain fell on the second morning." }),
      seg({
        block_id: "ch002_b0003",
        type: "dialogue",
        speaker: "bob",
        speaker_name: "Bob",
        text: '"Is anyone there?" Bob called.',
        confidence: 0.4,
      }),
    ],
    edit_warnings: [],
  };
}
const CH2_LOW = "/api/books/b1/chapters/2/segments?low_confidence=true";

/** A query client with the app's real retry default (main.tsx: retry 1), minus the backoff. */
function retryingClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: 1, retryDelay: 0, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
}

/** A route that answers only once the test lets it (`release`). */
function gated(server: MockApi, match: string, reply: () => Response) {
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  server.on("GET", match, async () => {
    await gate;
    return reply();
  });
  return () => release();
}

/** The always-mounted polite region "Next to check" speaks through. */
function walkRegion(): HTMLElement {
  const el = document.querySelector(".drainstrip + [aria-live='polite']");
  if (!(el instanceof HTMLElement)) throw new Error("no Next-to-check live region");
  return el;
}

describe("Review navigation", () => {
  afterEach(() => vi.restoreAllMocks());

  it("spoiler-safe is off by default, turned on from the View menu, and remembered for this viewer", async () => {
    const user = userEvent.setup();
    setupReview([alice(), bob(), mallory()]);
    const first = renderWithProviders(<Review />);

    // off: every character is named and no frontier control clutters the roster
    expect(await screen.findByText("Mallory")).toBeInTheDocument();
    expect(screen.queryByText(/enters ch 5/)).not.toBeInTheDocument();
    expect(screen.queryByRole("spinbutton", { name: "read through chapter" })).not.toBeInTheDocument();

    await turnOnSpoilerSafe(user);
    expect(screen.queryByText("Mallory")).not.toBeInTheDocument();
    expect(screen.getByText(/enters ch 5/)).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "read through chapter" })).toBeInTheDocument();
    expect(localStorage.getItem("seiyuu.review.spoilerSafe")).toBe("1");

    // a fresh visit starts masked
    first.unmount();
    setupReview([alice(), bob(), mallory()]);
    renderWithProviders(<Review />);
    expect(await screen.findByText(/enters ch 5/)).toBeInTheDocument();
    expect(screen.queryByText("Mallory")).not.toBeInTheDocument();

    // the menu shows the choice is on; turning it off is remembered too
    await user.click(screen.getByRole("button", { name: "View" }));
    const item = await screen.findByRole("menuitemcheckbox", { name: /Spoiler-safe/ });
    expect(item).toHaveAttribute("aria-checked", "true");
    await user.click(item);
    await user.keyboard("{Escape}");
    expect(await screen.findByText("Mallory")).toBeInTheDocument();
    expect(localStorage.getItem("seiyuu.review.spoilerSafe")).toBe("0");
  });

  it("a browser that refuses storage still gets a working page with spoiler-safe off", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("site data blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("site data blocked");
    });
    const user = userEvent.setup();
    setupReview([alice(), bob(), mallory()]);
    renderWithProviders(<Review />);

    expect(await screen.findByText("Mallory")).toBeInTheDocument();
    await turnOnSpoilerSafe(user); // works for this page even though it can't be remembered
    expect(screen.getByText(/enters ch 5/)).toBeInTheDocument();
  });

  it("with spoiler-safe on, the reassign and merge lists leave out characters not met yet", async () => {
    const user = userEvent.setup();
    setupReview([alice(), bob(), mallory()]);
    renderWithProviders(<Review />);
    expect(await screen.findByText("Mallory")).toBeInTheDocument();

    // off: everyone is offered
    await user.click(screen.getByRole("button", { name: "ALICE" }));
    await user.click(within(reassignPopover()).getByRole("button", { name: /Alice/ }));
    expect(await screen.findByRole("option", { name: "Mallory" })).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await user.click(within(reassignPopover()).getByRole("button", { name: "cancel" }));

    await turnOnSpoilerSafe(user);

    // reassign: Mallory is not a choice
    await user.click(screen.getByRole("button", { name: "ALICE" }));
    await user.click(within(reassignPopover()).getByRole("button", { name: /Alice/ }));
    expect(await screen.findByRole("option", { name: "Bob" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Mallory" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    await user.click(within(reassignPopover()).getByRole("button", { name: "cancel" }));

    // merge: nor there
    const aliceRow = screen.getByRole("row", { name: /Alice/ });
    await user.click(within(aliceRow).getByRole("button", { name: "✎" }));
    const dialog = await screen.findByRole("dialog", { name: /Edit character — Alice/ });
    await user.click(within(dialog).getByRole("button", { name: /— choose —/ }));
    expect(await screen.findByRole("option", { name: "Bob" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Mallory" })).not.toBeInTheDocument();
  });

  it("a line credited to a hidden character keeps that speaker in the reassign list without naming it", async () => {
    const user = userEvent.setup();
    const server = setupReview([alice(), bob(), mallory()]);
    const base = chapterOneSegments();
    server.get("/api/books/b1/chapters/1/segments", {
      ...base,
      segments: [
        ...base.segments,
        seg({
          block_id: "ch001_b0003",
          type: "dialogue",
          speaker: "mallory",
          speaker_name: "Mallory",
          text: '"Psst," someone whispered.',
          confidence: 0.9,
        }),
      ],
    } satisfies SegmentBrowserOut);
    renderWithProviders(<Review />);
    expect(await screen.findByText(/someone whispered/)).toBeInTheDocument();
    await turnOnSpoilerSafe(user);

    await user.click(screen.getByRole("button", { name: "▮▮▮▮▮" }));
    const pop = reassignPopover();
    expect(within(pop).getByRole("button", { name: /a hidden character/ })).toBeInTheDocument();
    expect(within(pop).queryByText(/Mallory/)).not.toBeInTheDocument();
  });

  it("?ch=&block=&seg= scrolls to that line and moves keyboard focus onto it", async () => {
    const scroll = vi.spyOn(Element.prototype, "scrollIntoView");
    setupReview();
    renderWithProviders(<Review />, { route: "/review?book=b1&ch=1&block=ch001_b0001&seg=1" });

    const line = await screen.findByText('"Hello there," said Alice.');
    await waitFor(() => expect(line).toHaveFocus());
    expect(line).toHaveClass("target");
    expect(scroll.mock.contexts).toContain(line);
    // the other line on the page is not marked
    expect(screen.getByText("It was a dark night.")).not.toHaveClass("target");
  });

  it("?block= alone opens the line's own chapter", async () => {
    const server = setupReview();
    server.get("/api/books/b1/chapters/2/segments", chapterTwoSegments());
    renderWithProviders(<Review />, { route: "/review?book=b1&block=ch002_b0003" });

    const line = await screen.findByText('"Is anyone there?" Bob called.');
    await waitFor(() => expect(line).toHaveFocus());
    expect(screen.getByRole("button", { name: /Ch 2 · The Middle/ })).toBeInTheDocument();
  });

  it("a link to a line that is no longer there says so plainly", async () => {
    setupReview();
    renderWithProviders(<Review />, { route: "/review?book=b1&ch=1&block=ch001_b0099" });
    expect(await screen.findByText(/The linked line isn't on this page/)).toBeInTheDocument();
    expect(screen.getByText("It was a dark night.")).toBeInTheDocument();
  });

  it("the chapter dropdown switches chapters and writes ?ch= into the URL", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/chapters/2/segments", chapterTwoSegments());
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1" },
    );

    await user.click(await screen.findByRole("button", { name: /Ch 1 · The Beginning/ }));
    await user.click(await screen.findByRole("option", { name: "Ch 2 · The Middle" }));

    expect(await screen.findByText("Rain fell on the second morning.")).toBeInTheDocument();
    expect(urlParams().get("ch")).toBe("2");
    expect(urlParams().get("book")).toBe("b1");
    expect(server.lastCall("GET", "/api/books/b1/chapters/2/segments")).toBeDefined();
  });

  it("Next to check finds the line in this chapter, then fetches the next chapter's lines to check", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/chapters/2/segments", chapterTwoSegments());
    server.get(CH2_LOW, { ...chapterTwoSegments(), segments: [chapterTwoSegments().segments[1]] });
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1" },
    );
    await screen.findByText("It was a dark night.");

    // first: the 0.55 Alice line on this page — no look-ahead request needed
    const next = screen.getByRole("button", { name: "Next to check ▸" });
    await user.click(next);
    const aliceLine = screen.getByText('"Hello there," said Alice.');
    await waitFor(() => expect(aliceLine).toHaveClass("target"));
    // the line is marked and announced, but focus stays on Next for the press-check-press loop
    expect(next).toHaveFocus();
    expect(screen.getByText('Next line to check, chapter 1: "Hello there," said Alice.')).toHaveClass("sr-only");
    expect(urlParams().get("block")).toBe("ch001_b0001");
    expect(urlParams().get("seg")).toBe("1");
    expect(server.lastCall("GET", "low_confidence")).toBeUndefined();

    // then: nothing left in chapter 1, so chapter 2's ?low_confidence= lines are fetched
    await user.click(next);
    const bobLine = await screen.findByText('"Is anyone there?" Bob called.');
    await waitFor(() => expect(bobLine).toHaveClass("target"));
    expect(screen.getByRole("button", { name: "Next to check ▸" })).toHaveFocus();
    expect(server.lastCall("GET", CH2_LOW)).toBeDefined();
    expect(urlParams().get("ch")).toBe("2");
    expect(urlParams().get("block")).toBe("ch002_b0003");
    expect(urlParams().get("seg")).toBe("0");
  });

  it("Next to check skips a chapter with nothing to read and stops at the end of the book in plain words", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.error("GET", CH2_LOW, 404, "not_found", "chapter 2 has no attribution");
    renderWithProviders(<Review />, {
      route: "/review?book=b1&ch=1&block=ch001_b0001&seg=1",
      queryClient: retryingClient(),
    });
    await screen.findByText('"Hello there," said Alice.');

    await user.click(screen.getByRole("button", { name: "Next to check ▸" }));
    expect(await screen.findByText(/Nothing left to check from here to the end of the book/)).toBeInTheDocument();
    expect(server.calls.filter((c) => c.url === CH2_LOW)).toHaveLength(1); // never retried
  });

  it("at the end of the book, 'start again' looks from the first chapter and lands there", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/chapters/2/segments", chapterTwoSegments());
    server.get("/api/books/b1/chapters/1/segments?low_confidence=true", {
      ...chapterOneSegments(),
      segments: [chapterOneSegments().segments[1]],
    });
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1&ch=2&block=ch002_b0003&seg=0" },
    );
    await screen.findByText('"Is anyone there?" Bob called.');

    // the last line to check in the last chapter: nothing further, and nothing fetched
    await user.click(screen.getByRole("button", { name: "Next to check ▸" }));
    expect(await screen.findByText(/Nothing left to check from here to the end of the book/)).toBeInTheDocument();
    expect(server.lastCall("GET", "low_confidence")).toBeUndefined();

    await user.click(screen.getByRole("button", { name: "start again from the first chapter" }));
    const aliceLine = await screen.findByText('"Hello there," said Alice.');
    await waitFor(() => expect(aliceLine).toHaveClass("target"));
    // the "start again" link went away with the note: focus moved to Next, not to <body>
    expect(screen.getByRole("button", { name: "Next to check ▸" })).toHaveFocus();
    expect(server.lastCall("GET", "/chapters/1/segments?low_confidence=true")).toBeDefined();
    expect(urlParams().get("ch")).toBe("1");
    expect(screen.queryByText(/Nothing left to check/)).not.toBeInTheDocument();
  });

  it("a failed look-ahead stops with the reason instead of retrying", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.error("GET", CH2_LOW, 500, "internal", "disk on fire");
    // the app's own default (main.tsx) retries a query once: the walk must opt out of that
    renderWithProviders(<Review />, {
      route: "/review?book=b1&ch=1&block=ch001_b0001&seg=1",
      queryClient: retryingClient(),
    });
    await screen.findByText('"Hello there," said Alice.');

    await user.click(screen.getByRole("button", { name: "Next to check ▸" }));
    expect(await screen.findByText(/Couldn't read chapter 2: disk on fire/)).toBeInTheDocument();
    expect(server.calls.filter((c) => c.url === CH2_LOW)).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Next to check ▸" })).toBeEnabled();
  });

  it("clicking a character shows only their lines, with a clear way back to every line", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/chapters/1/segments?speaker=alice", {
      ...chapterOneSegments(),
      segments: [chapterOneSegments().segments[1]],
    });
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1" },
    );
    expect(await screen.findByText("It was a dark night.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Alice" }));

    expect(await screen.findByText(/showing only/)).toHaveTextContent("showing only Alice · 1 line in this chapter");
    await waitFor(() => expect(screen.queryByText("It was a dark night.")).not.toBeInTheDocument());
    expect(screen.getByText('"Hello there," said Alice.')).toBeInTheDocument();
    expect(server.lastCall("GET", "/chapters/1/segments?speaker=alice")).toBeDefined();
    expect(urlParams().get("speaker")).toBe("alice");
    expect(screen.getByRole("button", { name: "Alice" })).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByRole("button", { name: "show all" }));
    expect(await screen.findByText("It was a dark night.")).toBeInTheDocument();
    expect(screen.queryByText(/showing only/)).not.toBeInTheDocument();
    expect(urlParams().get("speaker")).toBeNull();
  });

  it("a filtered chapter with none of their lines says so", async () => {
    const server = setupReview();
    server.get("/api/books/b1/chapters/1/segments?speaker=bob", { ...chapterOneSegments(), segments: [] });
    renderWithProviders(<Review />, { route: "/review?book=b1&speaker=bob" });
    expect(await screen.findByText("No lines for Bob in this chapter.")).toBeInTheDocument();
    expect(screen.getByText(/showing only/)).toHaveTextContent("showing only Bob · 0 lines in this chapter");
  });

  it("with spoiler-safe off, a reassign list opened before (or without) the character list names the speaker", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.error("GET", "/api/books/b1/characters", 500, "corrupt_artifact", "characters file unreadable");
    renderWithProviders(<Review />);
    expect(await screen.findByText("characters file unreadable")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "ALICE" }));
    const pop = reassignPopover();
    expect(within(pop).getByRole("button", { name: /Alice/ })).toBeInTheDocument();
    expect(within(pop).queryByText(/hidden/)).not.toBeInTheDocument();
  });

  it("Next to check follows the one-character filter: other speakers' lines are passed over", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    const bobLine = chapterTwoSegments().segments[1];
    server
      .get("/api/books/b1/chapters/2/segments", chapterTwoSegments())
      .get("/api/books/b1/chapters/1/segments?speaker=bob", { ...chapterOneSegments(), segments: [] })
      .get("/api/books/b1/chapters/2/segments?speaker=bob", { ...chapterTwoSegments(), segments: [bobLine] })
      .get("/api/books/b1/chapters/2/segments?speaker=bob&low_confidence=true", {
        ...chapterTwoSegments(),
        segments: [bobLine],
      });
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1&speaker=bob" },
    );
    expect(await screen.findByText("No lines for Bob in this chapter.")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Next to check ▸" })).not.toHaveAttribute("aria-disabled"));

    // Alice's 0.55 line in chapter 1 is not Bob's: the walk goes on to chapter 2, asking only for Bob
    await user.click(screen.getByRole("button", { name: "Next to check ▸" }));
    const line = await screen.findByText('"Is anyone there?" Bob called.');
    await waitFor(() => expect(line).toHaveClass("target"));
    expect(server.lastCall("GET", "/chapters/2/segments?speaker=bob&low_confidence=true")).toBeDefined();
    expect(server.lastCall("GET", "/chapters/1/segments?speaker=bob&low_confidence")).toBeUndefined();
    expect(urlParams().get("ch")).toBe("2");
    expect(urlParams().get("block")).toBe("ch002_b0003");
    expect(urlParams().get("speaker")).toBe("bob");
  });

  it("while Next to check looks ahead, the button keeps focus, says it is busy, and ignores more presses", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    const release = gated(server, CH2_LOW, () => errorResponse(404, "not_found", "chapter 2 has no attribution"));
    renderWithProviders(<Review />, { route: "/review?book=b1&ch=1&block=ch001_b0001&seg=1" });
    await screen.findByText('"Hello there," said Alice.');
    const region = walkRegion(); // mounted before anything is said into it
    expect(region).toBeEmptyDOMElement();

    const next = screen.getByRole("button", { name: "Next to check ▸" });
    next.focus();
    await user.keyboard("{Enter}");
    const busy = await screen.findByRole("button", { name: "looking in chapter 2…" });
    expect(busy).toBe(next);
    expect(busy).toHaveFocus();
    expect(busy).toHaveAttribute("aria-disabled", "true");
    expect(busy).toHaveAttribute("aria-busy", "true");
    expect(region).toHaveTextContent("Looking for the next line to check in chapter 2…");
    await user.keyboard("{Enter}"); // a second press while busy starts nothing
    expect(server.calls.filter((c) => c.url === CH2_LOW)).toHaveLength(1);

    release();
    await waitFor(() =>
      expect(region).toHaveTextContent("Nothing left to check from here to the end of the book."),
    );
    expect(screen.getByRole("button", { name: "Next to check ▸" })).toHaveFocus();
  });

  it("changing chapter while Next to check looks ahead abandons the search", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/chapters/2/segments", chapterTwoSegments());
    const release = gated(server, CH2_LOW, () =>
      jsonResponse({ ...chapterTwoSegments(), segments: [chapterTwoSegments().segments[1]] }),
    );
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1&ch=1&block=ch001_b0001&seg=1" },
    );
    await screen.findByText('"Hello there," said Alice.');

    await user.click(screen.getByRole("button", { name: "Next to check ▸" }));
    expect(await screen.findByRole("button", { name: "looking in chapter 2…" })).toBeInTheDocument();
    await waitFor(() => expect(server.lastCall("GET", CH2_LOW)).toBeDefined());

    await user.click(screen.getByRole("button", { name: /Ch 1 · The Beginning/ }));
    await user.click(await screen.findByRole("option", { name: "Ch 2 · The Middle" }));
    const bobLine = await screen.findByText('"Is anyone there?" Bob called.');
    expect(screen.getByRole("button", { name: "Next to check ▸" })).toBeInTheDocument();

    // the late answer lands nowhere: no jump to its line, no landing message
    release();
    await new Promise((r) => setTimeout(r, 30));
    expect(urlParams().get("ch")).toBe("2");
    expect(urlParams().get("block")).toBeNull();
    expect(bobLine).not.toHaveClass("target");
    expect(walkRegion()).toBeEmptyDOMElement();
    expect(screen.getByRole("button", { name: "Next to check ▸" })).toBeInTheDocument();
  });

  it("a live refresh after landing on a linked line does not scroll to it again", async () => {
    const scroll = vi.spyOn(Element.prototype, "scrollIntoView");
    const server = setupReview();
    const { queryClient } = renderWithProviders(<Review />, {
      route: "/review?book=b1&ch=1&block=ch001_b0001&seg=1",
    });
    const line = await screen.findByText('"Hello there," said Alice.');
    await waitFor(() => expect(line).toHaveFocus());
    const scrollsTo = () => scroll.mock.contexts.filter((c) => c === line).length;
    expect(scrollsTo()).toBe(1);
    const fetches = () => server.calls.filter((c) => c.url === "/api/books/b1/chapters/1/segments").length;
    const before = fetches();

    // what JobCompletionWatcher (or an edit) does: refresh every segment page
    await queryClient.invalidateQueries({ queryKey: ["segments"] });
    await waitFor(() => expect(fetches()).toBeGreaterThan(before));
    await new Promise((r) => setTimeout(r, 20));
    expect(scrollsTo()).toBe(1);
    expect(line).toHaveClass("target");
  });
});

describe("Review front matter", () => {
  /** b1 whose chapter 1 is front matter (no attribution) and chapter 2 is the story. */
  function frontMatterBook(server: MockApi, bookId = "b1") {
    server
      .error("GET", `/api/books/${bookId}/chapters/1/segments`, 404, "not_found", "chapter 1 has no attribution")
      .get(`/api/books/${bookId}/chapters/2/segments`, chapterTwoSegments());
  }

  it("an unattributed chapter 1 is skipped when no chapter was asked for, and ?ch=2 lands in the URL", async () => {
    const server = setupReview();
    frontMatterBook(server);
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1" },
    );
    expect(await screen.findByText("Rain fell on the second morning.")).toBeInTheDocument();
    expect(urlParams().get("ch")).toBe("2");
  });

  it("an explicit ?ch=1 stays on chapter 1 and says why it is empty", async () => {
    const server = setupReview();
    frontMatterBook(server);
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1&ch=1" },
    );
    expect(await screen.findByText("chapter 1 has no attribution")).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 20));
    expect(urlParams().get("ch")).toBe("1");
    expect(server.lastCall("GET", "/api/books/b1/chapters/2/segments")).toBeUndefined();
  });

  it("a link to a line in chapter 1 is not bounced to chapter 2", async () => {
    const server = setupReview();
    frontMatterBook(server);
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1&block=ch001_b0001" },
    );
    expect(await screen.findByText("chapter 1 has no attribution")).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 20));
    expect(urlParams().get("ch")).toBeNull();
    expect(urlParams().get("block")).toBe("ch001_b0001");
    expect(server.lastCall("GET", "/api/books/b1/chapters/2/segments")).toBeUndefined();
  });

  it("the skip happens again for the next book", async () => {
    const user = userEvent.setup();
    const server = mockApi().get("/api/books", {
      books: [
        makeBook({ book_id: "b1", title: "Book One", attributed: true }),
        makeBook({ book_id: "b2", title: "Book Two", attributed: true }),
      ],
    });
    registerBook(server, "b1", "Book One");
    registerBook(server, "b2", "Book Two");
    frontMatterBook(server, "b1");
    frontMatterBook(server, "b2");
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1" },
    );
    await screen.findByText("Rain fell on the second morning.");
    expect(urlParams().get("ch")).toBe("2");

    await user.click(screen.getByRole("button", { name: /Book One/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Book Two" }));
    await waitFor(() => expect(server.lastCall("GET", "/api/books/b2/chapters/2/segments")).toBeDefined());
    expect(urlParams().get("book")).toBe("b2");
    expect(urlParams().get("ch")).toBe("2");
  });
});

/* ------------------------------------------------------------------ power tools (1.10 B) */

describe("Review power tools", () => {
  afterEach(() => vi.restoreAllMocks());

  it("Looks right ✓ keeps the speaker (a same-speaker reassign) and the line leaves the queue", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    renderWithProviders(<Review />);

    const keep = await screen.findByRole("button", { name: "Looks right: keep Alice" });
    // only the line to check offers it
    expect(screen.getAllByRole("button", { name: /^Looks right/ })).toHaveLength(1);

    const confirmed = chapterOneSegments();
    confirmed.segments[1] = { ...confirmed.segments[1], confidence: 1 };
    server.get("/api/books/b1/chapters/1/segments", confirmed); // the refetch after the edit
    await user.click(keep);

    expect(server.jsonBodyOf("POST", "/api/books/b1/edits")).toEqual({
      op: "reassign",
      block_id: "ch001_b0001",
      segment_index: 1,
      speaker: "alice",
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: /^Looks right/ })).not.toBeInTheDocument());
    expect(screen.queryByText(/in review/)).not.toBeInTheDocument();
    // the button left with the line's "to check" state: focus is on that line's speaker chip
    await waitFor(() => expect(screen.getByRole("button", { name: "ALICE" })).toHaveFocus());
  });

  it("while one Looks right saves, the others wait, so a refusal lands under its own line", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    const base = chapterOneSegments();
    base.segments.push(
      seg({
        block_id: "ch001_b0002",
        type: "dialogue",
        speaker: "bob",
        speaker_name: "Bob",
        text: '"Who\'s there?" asked Bob.',
        confidence: 0.4,
      }),
    );
    server.get("/api/books/b1/chapters/1/segments", base);
    let release: () => void = () => undefined;
    server.on(
      "POST",
      "/api/books/b1/edits",
      () => new Promise((resolve) => (release = () => resolve(errorResponse(409, "render_active", "busy")))),
    );
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Looks right: keep Alice" }));
    const bobKey = screen.getByRole("button", { name: "Looks right: keep Bob" });
    await waitFor(() => expect(bobKey).toHaveAttribute("aria-disabled", "true"));
    await user.click(bobKey);
    expect(server.calls.filter((c) => c.method === "POST" && c.url === "/api/books/b1/edits")).toHaveLength(1);

    release();
    // the refusal is Alice's line's, shown under it
    const alert = await screen.findByRole("alert");
    expect(alert.closest(".margin")).toHaveTextContent("ALICE");
    expect(bobKey).not.toHaveAttribute("aria-disabled");
  });

  it("a refused Looks right says why in plain words and is not retried", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.error("POST", "/api/books/b1/edits", 409, "render_active", "a render job for 'b1' is running", {
      job_id: "job-r1",
      kind: "render",
      state: "running",
    });
    server.post("/api/jobs/job-r1/cancel", makeJob({ job_id: "job-r1", state: "canceled" }));
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "Looks right: keep Alice" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("A recording is running for this book; changes are paused until it finishes.");
    // the raw guard message is not the text; the recourse is right there in the line note
    expect(alert).not.toHaveTextContent("a render job for 'b1'");
    await user.click(within(alert).getByRole("button", { name: "Cancel recording" }));
    expect(server.calls.filter((c) => c.method === "POST" && c.url === "/api/jobs/job-r1/cancel")).toHaveLength(1);
    // the refused edit itself is never retried
    expect(server.calls.filter((c) => c.method === "POST" && c.url === "/api/books/b1/edits")).toHaveLength(1);
  });

  it("a confirmed unattributed quote reads as narration: no 'speaker unknown', no Looks right", async () => {
    const server = setupReview();
    const base = chapterOneSegments();
    server.get("/api/books/b1/chapters/1/segments", {
      ...base,
      segments: [
        seg({ block_id: "ch001_b0002", text: '"Who goes there?"', confidence: 1, unattributed_quote: true }),
        seg({ block_id: "ch001_b0003", text: '"Halt!"', confidence: 0, unattributed_quote: true }),
      ],
    } satisfies SegmentBrowserOut);
    renderWithProviders(<Review />);

    // the unconfirmed one keeps its marker and its Looks right (the narrator reads it)
    expect(await screen.findByRole("button", { name: "speaker unknown" })).toBeInTheDocument();
    // ...worded as what it records: the narrator reads this quote
    const keep = screen.getAllByRole("button", { name: "Narrator reads it: keep the narrator for this quote" });
    expect(keep).toHaveLength(1);
    expect(keep[0]).toHaveTextContent("Narrator reads it ✓");
    // the confirmed one is narration now
    expect(screen.getByRole("button", { name: "narration" })).toBeInTheDocument();
  });

  it("▶ on a recorded line plays its wav in the shared player; pressing it again pauses", async () => {
    const user = userEvent.setup();
    const play = vi.spyOn(HTMLMediaElement.prototype, "play");
    const pause = vi.spyOn(HTMLMediaElement.prototype, "pause");
    const server = setupReview();
    const base = chapterOneSegments();
    base.segments[1] = { ...base.segments[1], has_audio: true, audio_segment: 2, duration_seconds: 1.5, audio_key: "k9" };
    server.get("/api/books/b1/chapters/1/segments", base);
    renderWithProviders(<Review />);

    await screen.findByText('"Hello there," said Alice.');
    // only the line with audio gets a ▶
    const keys = screen.getAllByRole("button", { name: "play this line's recording" });
    expect(keys).toHaveLength(1);
    await user.click(keys[0]);
    await waitFor(() => expect(play).toHaveBeenCalled());
    const el = play.mock.contexts.at(-1) as HTMLAudioElement;
    expect(el.src).toContain("/api/books/b1/segments/ch001_b0001/audio?segment=2&v=k9");

    // jsdom never flips `paused` on play(); a real element would be playing now
    vi.spyOn(HTMLMediaElement.prototype, "paused", "get").mockReturnValue(false);
    await user.click(await screen.findByRole("button", { name: "pause this line's recording" }));
    expect(pause).toHaveBeenCalled();
    expect(await screen.findByRole("button", { name: "play this line's recording" })).toBeInTheDocument();
  });

  it("▶ on a line plays just that line even when a Listen chapter sits paused on it", async () => {
    const user = userEvent.setup();
    vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    const server = setupReview();
    const base = chapterOneSegments();
    base.segments[1] = { ...base.segments[1], has_audio: true, audio_segment: 2, duration_seconds: 1.5, audio_key: "k9" };
    server.get("/api/books/b1/chapters/1/segments", base);
    const src = "/api/books/b1/segments/ch001_b0001/audio?segment=2&v=k9";
    renderWithProviders(
      <>
        <Review />
        <PlayerProbe
          preload={{
            title: "The Beginning",
            clips: [
              { src, duration: 1.5, key: "ch001_b0001:2", speaker: "Alice", words: [] },
              { src: "/next.wav", duration: 2, key: "ch001_b0002:0", speaker: "narrator", words: [] },
            ],
          }}
        />
      </>,
    );

    await user.click(screen.getByRole("button", { name: "preload" }));
    expect(screen.getByTestId("player")).toHaveTextContent("2|The Beginning|Alice");
    await user.click(await screen.findByRole("button", { name: "play this line's recording" }));
    // the one-line playlist replaced the chapter: nothing runs on into the next line
    await waitFor(() => expect(screen.getByTestId("player")).toHaveTextContent("1|The Beginning · one line|Alice"));
  });

  it("▶ on a line by a hidden character keeps the name out of the player", async () => {
    const user = userEvent.setup();
    vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    const server = setupReview([alice(), bob(), mallory()]);
    const base = chapterOneSegments();
    base.segments[1] = {
      ...base.segments[1],
      speaker: "mallory",
      speaker_name: "Mallory",
      has_audio: true,
      audio_segment: 1,
      duration_seconds: 1.5,
    };
    server.get("/api/books/b1/chapters/1/segments", base);
    renderWithProviders(
      <>
        <Review />
        <PlayerProbe />
      </>,
    );
    await screen.findByText("Mallory");
    await turnOnSpoilerSafe(user);

    expect(await screen.findByRole("button", { name: "▮▮▮▮▮" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "play this line's recording" }));
    await waitFor(() => expect(screen.getByTestId("player")).toHaveTextContent("a hidden character"));
    expect(screen.getByTestId("player")).not.toHaveTextContent("Mallory");
  });

  it("confirmed quotes are counted as quotes with no speaker set, and don't keep Next to check busy", async () => {
    const server = setupReview();
    server.get("/api/books/b1/characters", {
      ...overviewFor("b1", [alice(), bob()]),
      low_confidence_segments: 0,
      unattributed_quote_segments: 2,
    });
    server.get("/api/books/b1/chapters/1/segments", {
      ...chapterOneSegments(),
      segments: [
        seg({ block_id: "ch001_b0002", text: '"Who goes there?"', confidence: 1, unattributed_quote: true }),
        seg({ block_id: "ch001_b0003", text: '"Halt!"', confidence: 1, unattributed_quote: true }),
      ],
    } satisfies SegmentBrowserOut);
    renderWithProviders(<Review />);

    expect(await screen.findByText(/2 quote\(s\) with no speaker set · 0 in this chapter/)).toBeInTheDocument();
    expect(screen.queryByText(/unattributed/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next to check ▸" })).toHaveAttribute("aria-disabled", "true");
  });

  it("thought lines are set in italics with a small thought tag", async () => {
    const server = setupReview();
    const base = chapterOneSegments();
    base.segments.push(
      seg({
        block_id: "ch001_b0002",
        type: "thought",
        speaker: "alice",
        speaker_name: "Alice",
        text: "She can't know.",
        confidence: 0.95,
      }),
    );
    server.get("/api/books/b1/chapters/1/segments", base);
    renderWithProviders(<Review />);

    expect(await screen.findByText("She can't know.")).toHaveClass("thought");
    expect(screen.getByText("thought")).toBeInTheDocument();
    expect(screen.getByText("It was a dark night.")).not.toHaveClass("thought");
  });

  it("the speaker and emotion chips open with Enter and Space", async () => {
    const user = userEvent.setup();
    setupReview();
    renderWithProviders(<Review />);

    const chip = await screen.findByRole("button", { name: "ALICE" });
    chip.focus();
    await user.keyboard("{Enter}");
    expect(await screen.findByText("who says this line?")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "cancel" }));

    screen.getByRole("button", { name: "angry ••" }).focus();
    await user.keyboard(" ");
    expect(await screen.findByRole("button", { name: "set emotion" })).toBeInTheDocument();
  });

  it("characters carry gender/age chips, and unknown gender warns about the default female voice", async () => {
    setupReview([
      { ...alice(), age_hint: "young adult" },
      { ...bob(), id: "pat", name: "Pat", gender: null },
    ]);
    renderWithProviders(<Review />);

    const aliceRow = await screen.findByRole("row", { name: /Alice/ });
    expect(within(aliceRow).getByText("female")).toBeInTheDocument();
    expect(within(aliceRow).getByText("young adult")).toBeInTheDocument();
    expect(within(aliceRow).queryByText(/automatic casting/)).not.toBeInTheDocument();

    const patRow = screen.getByRole("row", { name: /Pat/ });
    expect(within(patRow).getByText("gender unknown")).toBeInTheDocument();
    expect(
      within(patRow).getByText(
        "Gender unknown, so automatic casting gives them a female voice. If that's wrong, pick a voice for them by hand.",
      ),
    ).toBeInTheDocument();
  });

  it("a merge suggestion from the registry notes says what moves and merges with the existing edit", async () => {
    const user = userEvent.setup();
    const bobNoGender = { ...bob(), gender: null };
    const server = setupReview([alice(), bobNoGender]);
    server.get("/api/books/b1/characters", {
      ...overviewFor("b1", [alice(), bobNoGender]),
      notes: ["alias: adjudicator approved 'Bob' <-> 'Alice' at confidence 0.61 < 0.80 — flagged, not merged"],
    });
    renderWithProviders(<Review />);

    const cards = await screen.findByRole("region", { name: "possible duplicates" });
    expect(within(cards).getByText("Bob and Alice")).toBeInTheDocument();
    expect(within(cards).getByText('Moves 17 lines from Bob to Alice. "Bob" stays as another name for Alice.')).toBeInTheDocument();

    // flipping the direction updates the preview before anything is sent
    await user.click(within(cards).getByRole("button", { name: "keep Bob instead" }));
    expect(within(cards).getByText(/^Moves 42 lines from Alice to Bob/)).toBeInTheDocument();
    await user.click(within(cards).getByRole("button", { name: "keep Alice instead" }));

    server.get("/api/books/b1/characters", overviewFor("b1", [{ ...alice(), aliases: ["Ali", "Bob"], line_count: 59 }]));
    await user.click(within(cards).getByRole("button", { name: "Merge into Alice" }));
    expect(server.jsonBodyOf("POST", "/api/books/b1/edits")).toEqual({ op: "merge", loser_id: "bob", winner_id: "alice" });
    // after the merge the pair is one character, so the card is gone; focus lands on what happened
    const done = await within(cards).findByText("Merged Bob into Alice. No more suggestions.");
    await waitFor(() => expect(done).toHaveFocus());
    expect(within(cards).queryByText("Bob and Alice")).not.toBeInTheDocument();
  });

  it("a pair the server flagged as two different people (Mr. and Mrs.) never gets a merge card", async () => {
    const mr = { ...bob(), id: "mr-smith", name: "Mr. Smith", gender: null };
    const mrs = { ...alice(), id: "mrs-smith", name: "Mrs. Smith", gender: null };
    const server = setupReview([mr, mrs]);
    server.get("/api/books/b1/characters", {
      ...overviewFor("b1", [mr, mrs]),
      notes: [
        "alias: ambiguous 'smith' shared by ['Mr. Smith', 'Mrs. Smith'] — not merged, review",
        "not merging 'Mrs. Smith' with existing ['Mr. Smith'] (flagged for review, not auto-applied)",
      ],
    });
    renderWithProviders(<Review />);

    expect(await screen.findByRole("row", { name: /Mrs\. Smith/ })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "possible duplicates" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Merge into/ })).not.toBeInTheDocument();
  });

  it("Not the same person hides the card for this viewer; spoiler-safe hides cards naming hidden characters", async () => {
    const user = userEvent.setup();
    const people = [alice(), { ...bob(), gender: null }, { ...mallory(), gender: null }];
    const server = setupReview(people);
    server.get("/api/books/b1/characters", {
      ...overviewFor("b1", people),
      notes: [
        "alias: adjudicator approved 'Bob' <-> 'Alice' at confidence 0.61 < 0.80 — flagged, not merged",
        "not merging 'Mallory' with existing ['Alice'] (flagged for review, not auto-applied)",
      ],
    });
    renderWithProviders(<Review />);

    const cards = await screen.findByRole("region", { name: "possible duplicates" });
    expect(within(cards).getByText("Mallory and Alice")).toBeInTheDocument();
    await turnOnSpoilerSafe(user);
    expect(within(cards).queryByText(/Mallory/)).not.toBeInTheDocument();

    await user.click(within(cards).getByRole("button", { name: "Not the same person" }));
    expect(within(cards).queryByText("Bob and Alice")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(within(cards).getByText("Bob and Alice stay two people. No more suggestions.")).toHaveFocus(),
    );
    expect(JSON.parse(localStorage.getItem("seiyuu.review.notSame.b1") ?? "[]")).toEqual(["alice|bob"]);
    expect(server.lastCall("POST", "/api/books/b1/edits")).toBeUndefined();
  });

  it("the character editor's merge says how many lines move", async () => {
    const user = userEvent.setup();
    setupReview();
    renderWithProviders(<Review />);

    const bobRow = await screen.findByRole("row", { name: /Bob/ });
    await user.click(within(bobRow).getByRole("button", { name: "✎" }));
    const dialog = await screen.findByRole("dialog", { name: /Edit character — Bob/ });
    await user.click(within(dialog).getByRole("button", { name: /choose/ }));
    await user.click(await screen.findByRole("option", { name: "Alice" }));
    expect(within(dialog).getByText('Moves 17 lines from Bob to Alice. "Bob" stays as another name for Alice.')).toBeInTheDocument();
  });

  it("Go to line on a flagged paragraph opens and focuses it, in plain words", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/characters", {
      ...overviewFor("b1", [alice(), bob()]),
      flagged: [{ block_id: "ch001_b0001", chapter_index: 1, reason: "reconstruction mismatch" }],
    });
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1" },
    );

    expect(await screen.findByText(/couldn't work out who speaks in a paragraph here/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Go to line" }));
    const line = screen.getByText("It was a dark night.");
    await waitFor(() => expect(line).toHaveFocus());
    expect(urlParams().get("block")).toBe("ch001_b0001");
    expect(urlParams().get("ch")).toBe("1");
  });

  it("Go to line onto the line the link already names still moves focus there", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/characters", {
      ...overviewFor("b1", [alice(), bob()]),
      flagged: [{ block_id: "ch001_b0001", chapter_index: 1, reason: "reconstruction mismatch" }],
    });
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1&ch=1&block=ch001_b0001&seg=0" },
    );

    const line = await screen.findByText("It was a dark night.");
    await waitFor(() => expect(line).toHaveFocus());
    const go = screen.getByRole("button", { name: "Go to line" });
    go.focus();
    await user.click(go);
    await waitFor(() => expect(line).toHaveFocus());
    expect(urlParams().get("seg")).toBeNull();
  });

  it("an edit warning reads plainly and Go to line opens the line in its own chapter", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    server.get("/api/books/b1/chapters/2/segments", chapterTwoSegments());
    server.get("/api/books/b1/chapters/1/segments", {
      ...chapterOneSegments(),
      edit_warnings: [
        "reassign skipped: 'ch002_b0003'[0] is now different text than when the edit was recorded — re-record it if still wanted",
      ],
    });
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1&speaker=alice" },
    );

    expect(await screen.findByText(/A speaker fix no longer applies: the line changed since you made it/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Go to line" }));
    const line = await screen.findByText('"Is anyone there?" Bob called.');
    await waitFor(() => expect(line).toHaveFocus());
    expect(urlParams().get("ch")).toBe("2");
    expect(urlParams().get("seg")).toBe("0");
    // the one-character filter would hide the line: it is cleared
    expect(urlParams().get("speaker")).toBeNull();
  });

  it("History links each line change to its line and names hidden characters as hidden", async () => {
    const user = userEvent.setup();
    const server = setupReview([alice(), bob(), mallory()]);
    server.get("/api/books/b1/chapters/2/segments", chapterTwoSegments());
    server.get("/api/books/b1/edits", {
      version: 1,
      ops: [
        { op: "reassign", block_id: "ch002_b0003", segment_index: 0, speaker: "bob", text_anchor: '"Is anyone there?" Bob called.' },
        { op: "reassign", block_id: "ch001_b0001", segment_index: 1, speaker: "mallory", text_anchor: '"Hello there," said Alice.' },
      ],
    });
    renderWithProviders(
      <>
        <Review />
        <LocationProbe />
      </>,
      { route: "/review?book=b1" },
    );
    await screen.findByText("Mallory");
    await turnOnSpoilerSafe(user);

    await user.click(await screen.findByRole("button", { name: "History · 2" }));
    const drawer = await screen.findByRole("dialog", { name: "History" });
    expect(within(drawer).getByText('a hidden character says ""Hello there," said Alice."')).toBeInTheDocument();
    expect(within(drawer).queryByText(/Mallory/)).not.toBeInTheDocument();
    expect(within(drawer).getByText(/Only the newest change can be undone/)).toBeInTheDocument();
    // only the newest offers undo
    expect(within(drawer).getAllByRole("button", { name: "Undo this…" })).toHaveLength(1);

    const older = within(drawer).getByText('Bob says ""Is anyone there?" Bob called."').closest("li") as HTMLElement;
    await user.click(within(older).getByRole("button", { name: "Go to line" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "History" })).not.toBeInTheDocument());
    const line = await screen.findByText('"Is anyone there?" Bob called.');
    await waitFor(() => expect(line).toHaveFocus());
    expect(urlParams().get("ch")).toBe("2");
  });

  it("with spoiler-safe on, History hides the names in a rename or merge of a hidden character", async () => {
    const user = userEvent.setup();
    const server = setupReview([alice(), bob(), mallory()]);
    server.get("/api/books/b1/edits", {
      version: 1,
      ops: [
        { op: "rename", character_id: "mallory", new_name: "Mal", expected_name: "Mallory" },
        { op: "merge", loser_id: "mally", winner_id: "mallory", expected_loser_name: "Mally", expected_winner_name: "Mal" },
      ],
    });
    renderWithProviders(<Review />);
    await screen.findByText("Mallory");
    await turnOnSpoilerSafe(user);

    await user.click(await screen.findByRole("button", { name: "History · 2" }));
    const drawer = await screen.findByRole("dialog", { name: "History" });
    expect(within(drawer).getByText("Renamed a hidden character")).toBeInTheDocument();
    expect(within(drawer).getByText("Merged a name into a hidden character")).toBeInTheDocument();
    expect(within(drawer).queryByText(/Mal/)).not.toBeInTheDocument();
  });

  it("the undo question is pinned: a refetch that changes the newest closes it, and focus follows", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    const first = { op: "rename", character_id: "alice", new_name: "Alicia", expected_name: "Alice" };
    server.get("/api/books/b1/edits", { version: 1, ops: [first] });
    server.delete("/api/books/b1/edits/last", { removed: first });
    const { queryClient } = renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "History · 1" }));
    const drawer = await screen.findByRole("dialog", { name: "History" });
    await user.click(within(drawer).getByRole("button", { name: "Undo this…" }));
    await waitFor(() => expect(within(drawer).getByRole("button", { name: "Undo it" })).toHaveFocus());
    await user.click(within(drawer).getByRole("button", { name: "Keep it" }));
    await waitFor(() => expect(within(drawer).getByRole("button", { name: "Undo this…" })).toHaveFocus());

    await user.click(within(drawer).getByRole("button", { name: "Undo this…" }));
    // a background refetch brings in a newer change while the question is open
    server.get("/api/books/b1/edits", {
      version: 1,
      ops: [first, { op: "rename", character_id: "bob", new_name: "Robert", expected_name: "Bob" }],
    });
    await queryClient.invalidateQueries({ queryKey: ["edits", "b1"] });

    expect(await within(drawer).findByText(/The history changed since you opened it/)).toBeInTheDocument();
    expect(within(drawer).queryByRole("group", { name: "confirm undo" })).not.toBeInTheDocument();
    expect(server.lastCall("DELETE", "/api/books/b1/edits/last")).toBeUndefined();
    await waitFor(() => expect(within(drawer).getByRole("button", { name: "Undo this…" })).toHaveFocus());
  });

  it("undo refuses to remove a change it didn't show: a history that changed meanwhile is re-shown", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    const first = { op: "rename", character_id: "alice", new_name: "Alicia", expected_name: "Alice" };
    server.get("/api/books/b1/edits", { version: 1, ops: [first] });
    server.delete("/api/books/b1/edits/last", { removed: first });
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "History · 1" }));
    const drawer = await screen.findByRole("dialog", { name: "History" });
    await user.click(within(drawer).getByRole("button", { name: "Undo this…" }));

    // another tab recorded a newer change in the meantime
    server.get("/api/books/b1/edits", {
      version: 1,
      ops: [first, { op: "rename", character_id: "bob", new_name: "Robert", expected_name: "Bob" }],
    });
    await user.click(within(drawer).getByRole("button", { name: "Undo it" }));

    expect(await within(drawer).findByText(/The history changed since you opened it, so nothing was undone/)).toBeInTheDocument();
    expect(server.lastCall("DELETE", "/api/books/b1/edits/last")).toBeUndefined();
    expect(within(drawer).getByText('Renamed "Bob" to "Robert"')).toBeInTheDocument();
  });

  it("a refused undo is explained plainly and not retried", async () => {
    const user = userEvent.setup();
    const server = setupReview();
    const first = { op: "rename", character_id: "alice", new_name: "Alicia", expected_name: "Alice" };
    server.get("/api/books/b1/edits", { version: 1, ops: [first] });
    server.error("DELETE", "/api/books/b1/edits/last", 409, "render_active", "a render job for 'b1' is running");
    renderWithProviders(<Review />);

    await user.click(await screen.findByRole("button", { name: "History · 1" }));
    const drawer = await screen.findByRole("dialog", { name: "History" });
    await user.click(within(drawer).getByRole("button", { name: "Undo this…" }));
    await user.click(within(drawer).getByRole("button", { name: "Undo it" }));

    expect(await within(drawer).findByText("A recording is running for this book; changes are paused until it finishes.")).toBeInTheDocument();
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(1);
  });
});
