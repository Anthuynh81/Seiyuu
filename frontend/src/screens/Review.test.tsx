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
import { jsonResponse, mockApi, renderWithProviders } from "../test/utils";
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

    // distinguishable chip: "unattributed", never passed off as plain narration
    const chip = await screen.findByRole("button", { name: "unattributed" });
    // the drainstrip counts it: the book-level stat AND the chapter queue (0.0 + the 0.55 dialogue)
    expect(screen.getByText(/1 unattributed quote\(s\) · 2 in this chapter/)).toBeInTheDocument();
    // it wears the in-review marker like any low-confidence attribution
    expect(screen.getAllByText(/in review/)).toHaveLength(2);
    expect(screen.getByRole("button", { name: "next ▸" })).toBeEnabled();

    // its chip opens the reassign popover — the queue leads straight to the fix
    await user.click(chip);
    expect(await screen.findByText(/reassign · ch001_b0002 \[0\]/)).toBeInTheDocument();
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

    expect(await screen.findByText(/created 2 voice\(s\)/)).toBeInTheDocument();
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
    expect(screen.getByText(/anthropic cast is a paid call/)).toBeInTheDocument();
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

    expect(await screen.findByText(/created 2 voice\(s\)/)).toBeInTheDocument();
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
    expect(screen.getByRole("link", { name: /re-render in Render & Jobs/ })).toHaveAttribute(
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
    await user.click(screen.getByRole("checkbox", { name: /re-cast the 1 existing \(re-renders their audio\)/ }));
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
