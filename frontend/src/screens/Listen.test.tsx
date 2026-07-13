import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { BookDetail, RenderSummaryOut, SegmentBrowserOut, SegmentRow, SegmentWords } from "../api/types";
import { TransportBar } from "../app/TransportBar";
import { makeBook } from "../test/fixtures";
import { mockApi, renderWithProviders } from "../test/utils";
import { Listen } from "./Listen";

/** Text "a bb ccc" over a 90s clip gives DISTINCT interpolated vs whisper offsets for "bb":
    weights 2/3/4 (length+1) interpolate it to 20s, while the whisper fixture below puts it
    at 70s — so the transport's elapsed readout proves which timing source is live. */
function makeRow(overrides: Partial<SegmentRow> = {}): SegmentRow {
  return {
    block_id: "blk1",
    segment_index: 0,
    type: "narration",
    speaker: null,
    speaker_name: null,
    text: "a bb ccc",
    confidence: 0.95,
    unattributed_quote: false,
    has_audio: true,
    audio_segment: 0,
    duration_seconds: 90,
    voice_id: "v-narr",
    audio_key: "k1",
    ...overrides,
  };
}

const detail: BookDetail = {
  status: makeBook({ book_id: "b1", title: "Whale Story", attributed: true, rendered: true }),
  chapters: [
    { index: 1, title: "Loomings", blocks: 3, speakable_blocks: 3 },
    { index: 2, title: "The Carpet-Bag", blocks: 2, speakable_blocks: 2 },
    { index: 3, title: "The Spouter-Inn", blocks: 2, speakable_blocks: 2 },
  ],
  runtime_estimate_seconds: null,
  active_job: null,
  recent_jobs: [],
  downloads: { m4b: null, chapter_mp3s: [] },
  cover: null,
};

const summary: RenderSummaryOut = {
  book_id: "b1",
  mode: "multivoice",
  chapters: [
    { index: 1, title: "Loomings", segments: 1, duration_seconds: 90 },
    { index: 2, title: "The Carpet-Bag", segments: 1, duration_seconds: 30 },
  ],
  total_seconds: 120,
  voices_used: { "v-narr": { engine: "kokoro", engine_model_version: "1.0", kind: "preset" } },
  validation_failures: 0,
  rendered_assignment_hash: null,
  active_mode: "multi",
  available_modes: ["multi"],
};

const ch1: SegmentBrowserOut = {
  chapter_index: 1,
  title: "I. Loomings", // the title handed to the player (distinct from the toc title on purpose)
  segments: [
    makeRow(),
    makeRow({
      block_id: "blk9",
      type: "dialogue",
      speaker: "ishmael",
      speaker_name: "Ishmael",
      text: "Call me Ishmael.",
      has_audio: false,
      audio_segment: null,
      duration_seconds: null,
      voice_id: null,
      audio_key: null,
    }),
  ],
  edit_warnings: [],
};

const ch2: SegmentBrowserOut = {
  chapter_index: 2,
  title: "II. The Carpet-Bag",
  segments: [makeRow({ block_id: "blk2", text: "the ship sailed", duration_seconds: 30, audio_key: "k2" })],
  edit_warnings: [],
};

/** The whole GET surface one rendered book needs. Word-timing routes default to 404 so the
    read-along sits on interpolation; a test overrides them (latest registration wins). */
function serveBook() {
  return mockApi()
    .get("/api/books", {
      books: [
        makeBook({ book_id: "b1", title: "Whale Story", attributed: true, rendered: true }),
        makeBook({ book_id: "b-fresh", title: "Fresh One" }),
      ],
    })
    .get("/api/books/b1", detail)
    .get("/api/books/b1/render", summary)
    .get("/api/books/b1/chapters/1/segments", ch1)
    .get("/api/books/b1/chapters/2/segments", ch2)
    .error("GET", "/api/books/b1/segments/blk1/words?segment=0", 404, "not_found", "no word timings")
    .error("GET", "/api/books/b1/segments/blk2/words?segment=0", 404, "not_found", "no word timings");
}

/** The reader hands clips to the player context; TransportBar (mounted the way main.tsx does)
    is how that handoff becomes observable — it only wakes on /listen with clips loaded. */
function renderListen(route = "/listen?book=b1") {
  return renderWithProviders(
    <>
      <Listen />
      <TransportBar />
    </>,
    { route },
  );
}

describe("Listen", () => {
  it("with no ?book the shelf lists books, disables unrendered ones, and picking a tile opens the reader", async () => {
    const user = userEvent.setup();
    serveBook();
    const { container } = renderListen("/listen");

    // a book with no audio cannot be opened
    const fresh = await screen.findByRole("button", { name: /not rendered/ });
    expect(fresh).toBeDisabled();

    // cover art that fails to load degrades to a title/author tile (jsdom never loads imgs,
    // so fire the error by hand — that IS the degradation path)
    const cover = container.querySelector<HTMLImageElement>('img[src="/api/books/b1/cover"]');
    expect(cover).not.toBeNull();
    fireEvent.error(cover!);
    await user.click(await screen.findByRole("button", { name: /Whale Story/ }));

    // ?book= is set: the reader replaces the shelf
    expect(await screen.findByRole("heading", { name: "Whale Story" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /not rendered/ })).not.toBeInTheDocument();
  });

  it("?book= deep-links into the reader: chapter text, speaker margins, and render provenance", async () => {
    serveBook();
    renderListen();

    expect(await screen.findByRole("heading", { name: "Whale Story" })).toBeInTheDocument();
    // playable rows are split into clickable word spans; non-playable rows keep their plain text
    expect(await screen.findByText("bb")).toBeInTheDocument();
    expect(screen.getByText("Call me Ishmael.")).toBeInTheDocument();
    // margins credit the speaker (uppercased) and the voice that rendered the row
    expect(screen.getByText("ISHMAEL")).toBeInTheDocument();
    expect(screen.getAllByText("narration").length).toBeGreaterThan(0);
    expect(screen.getByText("v-narr")).toBeInTheDocument();
    expect(screen.getByText(/multivoice · 1 voices/)).toBeInTheDocument();
    // the header sub names the current chapter from the book detail
    expect(screen.getByText(/Loomings — click any word/)).toBeInTheDocument();
  });

  it("loading a chapter hands its clips to the player: the transport wakes paused with title and total time", async () => {
    const user = userEvent.setup();
    serveBook();
    renderListen();

    await screen.findByRole("button", { name: "play/pause" });
    expect(screen.getByText("paused")).toBeInTheDocument(); // no autoplay on a fresh open
    expect(screen.getByText("I. Loomings")).toBeInTheDocument(); // player got the segments title
    expect(screen.getByText("0:00 / 1:30")).toBeInTheDocument(); // one 90s clip

    await user.click(screen.getByRole("button", { name: "play/pause" }));
    expect(await screen.findByText("playing")).toBeInTheDocument();
  });

  it("the contents menu disables unrendered chapters and selecting one fetches + shows that chapter", async () => {
    const user = userEvent.setup();
    const server = serveBook();
    renderListen();

    await screen.findByText("bb"); // chapter 1 is on the page first
    await user.click(screen.getByRole("button", { name: /contents/ }));
    // chapter 3 is in the book but absent from the render summary
    expect(screen.getByRole("button", { name: /The Spouter-Inn/ })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: /The Carpet-Bag/ }));
    expect(await screen.findByText("sailed")).toBeInTheDocument();
    expect(screen.getByText(/The Carpet-Bag — click any word/)).toBeInTheDocument();
    expect(server.lastCall("GET", "/chapters/2/segments")).toBeDefined();
    // the menu closed on selection
    expect(screen.queryByRole("button", { name: /The Spouter-Inn/ })).not.toBeInTheDocument();
  });

  it("whisper word timings replace interpolation: clicking a word seeks the player to its spoken time", async () => {
    const server = serveBook();
    const whisper: SegmentWords = {
      words: [
        { start: 0, end: 5, word: "a" },
        { start: 70, end: 80, word: "bb" },
        { start: 80, end: 90, word: "ccc" },
      ],
      audio_duration: 90,
      source: "whisper",
    };
    server.get("/api/books/b1/segments/blk1/words?segment=0", whisper); // overrides the 404 default
    renderListen();

    await screen.findByRole("button", { name: "play/pause" });
    // interpolation would put "bb" at 0:20; whisper puts it at 1:10. The word spans are
    // imperative DOM (not React) and timings land asynchronously with no other visible
    // signal, so re-click via fireEvent inside waitFor until the whisper pass has applied.
    await waitFor(() => {
      fireEvent.click(screen.getByText("bb"));
      expect(screen.getByText("1:10 / 1:30")).toBeInTheDocument();
    });
    expect(screen.getByText("playing")).toBeInTheDocument();
  });

  it("a 404 from segment-words degrades to interpolated word timing with no error state", async () => {
    const user = userEvent.setup();
    const server = serveBook(); // word routes 404 by default
    renderListen();

    await screen.findByRole("button", { name: "play/pause" });
    await waitFor(() => expect(server.lastCall("GET", "words?segment=0")).toBeDefined());
    // the reader stays intact — no refusal banner of either flavor
    expect(screen.queryByText(/not attributed/)).not.toBeInTheDocument();
    expect(screen.queryByText(/no audio yet/)).not.toBeInTheDocument();

    // clicking a word still seeks — to the length-interpolated offset (weights 2/3/4 over 90s)
    await user.click(screen.getByText("bb"));
    expect(await screen.findByText("0:20 / 1:30")).toBeInTheDocument();
    expect(screen.getByText("playing")).toBeInTheDocument();
  });

  it("a chapter with no rendered audio shows the render nudge and never loads the player", async () => {
    const server = serveBook();
    server.get("/api/books/b1/chapters/1/segments", {
      ...ch1,
      segments: [
        makeRow({ has_audio: false, audio_segment: null, duration_seconds: null, voice_id: null, audio_key: null }),
      ],
    });
    renderListen();

    expect(await screen.findByText(/this chapter has no audio yet/)).toBeInTheDocument();
    // no clips were handed over: the transport stays on the job console
    expect(screen.getByText(/console idle/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "play/pause" })).not.toBeInTheDocument();
  });

  it("reading prefs (page theme + text size) merge and persist under seiyuu.reading", async () => {
    const user = userEvent.setup();
    serveBook();
    renderListen();

    await screen.findByRole("heading", { name: "Whale Story" });
    await user.click(screen.getByRole("button", { name: "sepia" }));
    await user.click(screen.getByRole("button", { name: "A++" }));
    expect(JSON.parse(localStorage.getItem("seiyuu.reading")!)).toEqual({ theme: "sepia", size: "l" });

    // changing one pref never clobbers the other
    await user.click(screen.getByRole("button", { name: "dark" }));
    expect(JSON.parse(localStorage.getItem("seiyuu.reading")!)).toEqual({ theme: "dark", size: "l" });
  });
});
