import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { useLocation } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import type {
  BookDetail,
  RenderSummaryOut,
  SegmentBrowserOut,
  SegmentRow,
  SegmentWords,
  VoiceAssignment,
  VoiceOut,
} from "../api/types";
import { TransportBar } from "../app/TransportBar";
import { usePlayer } from "../app/usePlayer";
import { makeBook } from "../test/fixtures";
import { jsonResponse, mockApi, renderWithProviders } from "../test/utils";
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

function makeVoice(voice_id: string, name: string): VoiceOut {
  return {
    voice_id,
    name,
    kind: "preset",
    engine: "kokoro",
    preset_id: "af_heart",
    blend: null,
    reference_audio: null,
    seed: 1,
    consent_attested: false,
    consent: null,
    tags: [],
    created_at: "2026-10-01",
    has_audition: false,
  };
}

const voices = {
  voices: [makeVoice("v-narr", "Warm Narrator"), makeVoice("v-ish", "Sea Captain")],
  unreadable: [],
};

const cast: VoiceAssignment = {
  schema_version: 1,
  book_id: "b1",
  stage: "final",
  narrator_voice_id: "v-narr",
  assignments: { ishmael: "v-ish" },
  thought_voice_id: null,
  created_at: "2026-10-01",
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
    .get("/api/voices", voices)
    .get("/api/books/b1/assignment", cast)
    .error("GET", "/api/books/b1/chapters/1/words", 404, "not_found", "no word timings")
    .error("GET", "/api/books/b1/chapters/2/words", 404, "not_found", "no word timings");
}

/** The reader hands clips to the player context; TransportBar (mounted the way main.tsx does)
    is how that handoff becomes observable — it only wakes on /listen with clips loaded. */
function Where() {
  const loc = useLocation();
  return <output data-testid="where">{loc.pathname + loc.search}</output>;
}

function renderListen(route = "/listen?book=b1") {
  return renderWithProviders(
    <>
      <Listen />
      <TransportBar />
      <Where />
    </>,
    { route },
  );
}

/** Leaving Listen for a fix and coming back: the player (in App, above the routes) keeps
    playing while Listen unmounts. "away" unmounts Listen; pressing it again brings it back. */
function RoundTrip({ startAway = false }: { startAway?: boolean }) {
  const [away, setAway] = useState(startAway);
  const player = usePlayer();
  // what Review does when it plays one line: a one-clip list for the same book
  const oneLine = [
    { src: "/api/books/b1/segments/blk9/audio?segment=0", duration: 4, key: "blk9:0", speaker: "ishmael", words: [] },
  ];
  return (
    <>
      {!away && <Listen />}
      <button onClick={() => setAway((v) => !v)}>away</button>
      <button onClick={() => player?.load("b1", "one line", oneLine, { autoplay: true })}>play one line</button>
      <output data-testid="player">{`${player?.playing ? "on" : "off"}|${player?.clips.map((c) => c.key).join(",")}`}</output>
      <TransportBar />
    </>
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

  it("with no ?book it opens the remembered book once, and '‹ shelf' still goes back to the shelf", async () => {
    const user = userEvent.setup();
    serveBook();
    localStorage.setItem("seiyuu.currentBook", "b1");
    renderListen("/listen");
    expect(await screen.findByRole("heading", { name: "Whale Story" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "‹ shelf" }));
    // not bounced straight back into the remembered book
    expect(await screen.findByRole("button", { name: /not rendered/ })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Whale Story" })).not.toBeInTheDocument();
  });

  it("an empty shelf nudges with a real Add-a-book link", async () => {
    mockApi().get("/api/books", { books: [] });
    renderListen("/listen");
    expect(await screen.findByRole("link", { name: "Add a book" })).toHaveAttribute("href", "/#add");
  });

  it("a deep link to a book with no audio yet shows its next step instead of a loading line", async () => {
    const fresh = makeBook({ book_id: "b-fresh", title: "Fresh One", attributed: true, assigned: true });
    mockApi()
      .get("/api/books", { books: [fresh] })
      .get("/api/books/b-fresh", { ...detail, status: fresh });
    renderListen("/listen?book=b-fresh");
    expect(await screen.findByRole("link", { name: "Record the audiobook" })).toHaveAttribute(
      "href",
      "/render?book=b-fresh#record",
    );
    expect(screen.queryByText("setting the page…")).not.toBeInTheDocument();
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
    // the recording voice by its library NAME, never the raw id
    expect(await screen.findByText("Warm Narrator")).toBeInTheDocument();
    expect(screen.queryByText("v-narr")).not.toBeInTheDocument();
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
    server.get("/api/books/b1/chapters/1/words", {
      book_id: "b1",
      chapter: 1,
      words: { "blk1:0": whisper },
    }); // overrides the 404 default
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
    await waitFor(() => expect(server.lastCall("GET", "chapters/1/words")).toBeDefined());
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

    expect(await screen.findByText(/This chapter has no audio yet/)).toBeInTheDocument();
    // the nudge is a real link to the recording controls, not prose naming a screen
    expect(screen.getByRole("link", { name: "Record this chapter" })).toHaveAttribute(
      "href",
      "/render?book=b1#record",
    );
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
  describe("links back to the fixes", () => {
    it("the speaker chip's menu: Fix this line lands on that line in Review", async () => {
      const user = userEvent.setup();
      serveBook();
      renderListen();

      await user.click(await screen.findByRole("button", { name: "Ishmael: fix this line" }));
      await user.click(await screen.findByRole("menuitem", { name: /Fix this line/ }));
      expect(screen.getByTestId("where")).toHaveTextContent("/review?book=b1&ch=1&block=blk9&seg=0");
    });

    it("Open in Review opens the chapter", async () => {
      const user = userEvent.setup();
      serveBook();
      renderListen();

      await user.click(await screen.findByRole("button", { name: "Ishmael: fix this line" }));
      await user.click(await screen.findByRole("menuitem", { name: /Open in Review/ }));
      expect(screen.getByTestId("where")).toHaveTextContent("/review?book=b1&ch=1");
      expect(screen.getByTestId("where")).not.toHaveTextContent("block=");
    });

    it("Fix pronunciation carries the word last clicked in that line to the dictionary", async () => {
      const user = userEvent.setup();
      serveBook();
      renderListen();

      await screen.findByRole("button", { name: "play/pause" });
      await user.click(screen.getByText("bb")); // also plays from that word
      await user.click(screen.getByRole("button", { name: "narration: fix this line" }));
      await user.click(await screen.findByRole("menuitem", { name: /Fix pronunciation of “bb”/ }));
      expect(screen.getByTestId("where")).toHaveTextContent("/lexicon?book=b1&term=bb");
    });

    it("Fix pronunciation uses the text selected in that line, else opens the dictionary plain", async () => {
      const user = userEvent.setup();
      serveBook();
      renderListen();

      const quote = await screen.findByText("Call me Ishmael.");
      const range = document.createRange();
      range.setStart(quote.firstChild!, 8);
      range.setEnd(quote.firstChild!, 16); // "Ishmael."
      window.getSelection()!.removeAllRanges();
      window.getSelection()!.addRange(range);
      await user.click(screen.getByRole("button", { name: "Ishmael: fix this line" }));
      expect(await screen.findByRole("menuitem", { name: /Fix pronunciation of “Ishmael”/ })).toBeInTheDocument();
      await user.keyboard("{Escape}");

      // a selection in ANOTHER line is not this line's word
      await user.click(screen.getByRole("button", { name: "narration: fix this line" }));
      const plain = await screen.findByRole("menuitem", { name: /Fix pronunciation/ });
      expect(plain).not.toHaveTextContent("“");
      await user.click(plain);
      expect(screen.getByTestId("where")).toHaveTextContent(/^\/lexicon\?book=b1$/);
    });
  });

  describe("voices in the margin", () => {
    it("names the recording voice, notes a line whose voice changed since, and never shows raw ids", async () => {
      const server = serveBook();
      server.get("/api/books/b1/chapters/1/segments", {
        ...ch1,
        segments: [
          makeRow(), // narration, recorded with the narrator: unchanged
          // Ishmael now has his own voice, but this line was recorded with the narrator's
          makeRow({
            block_id: "blk5",
            type: "dialogue",
            speaker: "ishmael",
            speaker_name: "Ishmael",
            text: "Aye.",
            audio_key: "k5",
          }),
          // recorded with a voice since deleted from the library
          makeRow({ block_id: "blk6", text: "gone voice", voice_id: "v-deleted", audio_key: "k6" }),
        ],
      });
      renderListen();

      expect(await screen.findByText("changed since recording — now Sea Captain")).toBeInTheDocument();
      // the deleted voice's narration line would be the narrator's today: flagged too
      expect(screen.getByText("changed since recording — now Warm Narrator")).toBeInTheDocument();
      expect(screen.getAllByText("Warm Narrator")).toHaveLength(2);
      expect(screen.getByText("a voice no longer in your library")).toBeInTheDocument();
      expect(screen.queryByText("v-narr")).not.toBeInTheDocument();
      expect(screen.queryByText("v-deleted")).not.toBeInTheDocument();
    });

    it("a one-narrator recording never flags lines as changed", async () => {
      const server = serveBook();
      server.get("/api/books/b1/render", {
        ...summary,
        mode: "single",
        active_mode: "single",
        available_modes: ["single"],
      });
      server.get("/api/books/b1/chapters/1/segments", {
        ...ch1,
        segments: [makeRow({ type: "dialogue", speaker: "ishmael", speaker_name: "Ishmael" })],
      });
      renderListen();

      expect(await screen.findByText("Warm Narrator")).toBeInTheDocument();
      expect(screen.queryByText(/changed since recording/)).not.toBeInTheDocument();
    });

    it("a voice whose files can't be read is not called deleted; it links to Voices", async () => {
      const server = serveBook();
      server.get("/api/voices", { ...voices, unreadable: [{ voice_id: "v-broken", error: "bad meta.json" }] });
      server.get("/api/books/b1/chapters/1/segments", { ...ch1, segments: [makeRow({ voice_id: "v-broken" })] });
      renderListen();

      const link = await screen.findByRole("link", { name: "a voice that can't be opened right now" });
      expect(link).toHaveAttribute("href", "/voices");
      expect(screen.queryByText("a voice no longer in your library")).not.toBeInTheDocument();
    });

    it("when the voice library can't be loaded, the raw voice id is still credited", async () => {
      const server = serveBook();
      server.error("GET", "/api/voices", 500, "internal", "boom");
      renderListen();

      expect(await screen.findByText("v-narr")).toBeInTheDocument();
      expect(screen.queryByText("a voice no longer in your library")).not.toBeInTheDocument();
    });
  });

  describe("resume where you left off", () => {
    it("offers the remembered chapter and time, and resuming plays from there", async () => {
      const user = userEvent.setup();
      serveBook();
      localStorage.setItem("seiyuu.listen.b1", JSON.stringify({ chapter: 2, clip: "blk2:0", offset: 12, seconds: 12 }));
      renderListen();

      const resume = await screen.findByRole("button", { name: "Resume chapter 2 at 0:12" });
      expect(screen.getByText(/where you stopped · The Carpet-Bag/)).toBeInTheDocument();
      await user.click(resume);

      expect(await screen.findByText("sailed")).toBeInTheDocument();
      expect(await screen.findByText("0:12 / 0:30")).toBeInTheDocument();
      expect(screen.getByText("playing")).toBeInTheDocument();
      // offered once: gone after use
      expect(screen.queryByRole("button", { name: /Resume chapter/ })).not.toBeInTheDocument();
    });

    it("remembers the place while playing, never the paused 0:00 a fresh open loads at", async () => {
      const user = userEvent.setup();
      serveBook();
      const saved = JSON.stringify({ chapter: 2, clip: "blk2:0", offset: 12, seconds: 12 });
      localStorage.setItem("seiyuu.listen.b1", saved);
      renderListen();

      await screen.findByRole("button", { name: "play/pause" }); // chapter 1 loaded, paused
      expect(screen.getByRole("button", { name: "Resume chapter 2 at 0:12" })).toBeInTheDocument();
      expect(localStorage.getItem("seiyuu.listen.b1")).toBe(saved);

      await user.click(screen.getByText("bb")); // plays from 0:20 in chapter 1
      await waitFor(() =>
        expect(JSON.parse(localStorage.getItem("seiyuu.listen.b1")!)).toEqual({
          chapter: 1,
          clip: "blk1:0",
          offset: 20,
          seconds: 20,
        }),
      );
      // listening started here, so the old offer goes away
      expect(screen.queryByRole("button", { name: /Resume chapter/ })).not.toBeInTheDocument();
    });

    it("no offer for a garbled record or the very start of the book", async () => {
      serveBook();
      localStorage.setItem("seiyuu.listen.b1", JSON.stringify({ chapter: 1, clip: "blk1:0", offset: 1, seconds: 1 }));
      const first = renderListen();
      await screen.findByRole("button", { name: "play/pause" });
      expect(screen.queryByRole("button", { name: /Resume chapter/ })).not.toBeInTheDocument();
      first.unmount();

      serveBook();
      localStorage.setItem("seiyuu.listen.b1", "{garbled");
      renderListen();
      await screen.findByRole("button", { name: "play/pause" });
      expect(screen.queryByRole("button", { name: /Resume chapter/ })).not.toBeInTheDocument();
    });

    it("blocked storage: the reader still works, with nothing to resume", async () => {
      const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
        throw new DOMException("blocked", "SecurityError");
      });
      try {
        serveBook();
        renderListen();
        expect(await screen.findByText("bb")).toBeInTheDocument();
        expect(screen.queryByRole("button", { name: /Resume chapter/ })).not.toBeInTheDocument();
      } finally {
        getItem.mockRestore();
      }
    });

    it("resuming in the chapter already open seeks there without switching chapter", async () => {
      const user = userEvent.setup();
      serveBook();
      localStorage.setItem("seiyuu.listen.b1", JSON.stringify({ chapter: 1, clip: "blk1:0", offset: 20, seconds: 20 }));
      renderListen();

      await screen.findByRole("button", { name: "play/pause" }); // chapter 1 loaded, paused
      await user.click(screen.getByRole("button", { name: "Resume chapter 1 at 0:20" }));
      expect(await screen.findByText("0:20 / 1:30")).toBeInTheDocument();
      expect(screen.getByText("playing")).toBeInTheDocument();
      expect(screen.getByText("bb")).toBeInTheDocument(); // still chapter 1
    });

    it("leaving for a fix and coming back keeps playing the same chapter, with nothing to resume", async () => {
      const user = userEvent.setup();
      serveBook();
      renderWithProviders(<RoundTrip />, { route: "/listen?book=b1" });

      await screen.findByRole("button", { name: "play/pause" });
      await user.click(screen.getByTitle("next rendered chapter"));
      await user.click(await screen.findByText("sailed"));
      await waitFor(() => expect(screen.getByTestId("player")).toHaveTextContent("on|blk2:0"));
      await waitFor(() => expect(JSON.parse(localStorage.getItem("seiyuu.listen.b1")!).chapter).toBe(2));

      await user.click(screen.getByRole("button", { name: "away" })); // off to Review...
      expect(screen.queryByText("sailed")).not.toBeInTheDocument();
      expect(screen.getByTestId("player")).toHaveTextContent("on|blk2:0");
      await user.click(screen.getByRole("button", { name: "away" })); // ...and back

      expect(await screen.findByText("sailed")).toBeInTheDocument();
      expect(screen.queryByText("bb")).not.toBeInTheDocument(); // not rewound to chapter 1
      expect(screen.getByTestId("player")).toHaveTextContent("on|blk2:0");
      expect(screen.queryByRole("button", { name: /Resume chapter/ })).not.toBeInTheDocument();
    });

    it("a line played from Review does not count as listening: the offer stays", async () => {
      const user = userEvent.setup();
      serveBook();
      localStorage.setItem("seiyuu.listen.b1", JSON.stringify({ chapter: 2, clip: "blk2:0", offset: 12, seconds: 12 }));
      renderWithProviders(<RoundTrip startAway />, { route: "/listen?book=b1" });

      await user.click(screen.getByRole("button", { name: "play one line" }));
      expect(screen.getByTestId("player")).toHaveTextContent("on|blk9:0");
      await user.click(screen.getByRole("button", { name: "away" })); // open Listen

      expect(await screen.findByRole("button", { name: "Resume chapter 2 at 0:12" })).toBeInTheDocument();
      // the one-liner was not remembered as the place either
      expect(JSON.parse(localStorage.getItem("seiyuu.listen.b1")!).chapter).toBe(2);
    });

    it("picking another chapter cancels a resume still waiting to load", async () => {
      const user = userEvent.setup();
      const server = serveBook();
      let releaseCh2: () => void = () => {};
      const ch2Ready = new Promise<void>((r) => (releaseCh2 = r));
      server.on("GET", "/api/books/b1/chapters/2/segments", async () => {
        await ch2Ready;
        return jsonResponse(ch2);
      });
      localStorage.setItem("seiyuu.listen.b1", JSON.stringify({ chapter: 2, clip: "blk2:0", offset: 12, seconds: 12 }));
      renderListen();

      await screen.findByRole("button", { name: "play/pause" });
      await user.click(screen.getByRole("button", { name: "Resume chapter 2 at 0:12" }));
      await user.click(screen.getByTitle("previous rendered chapter")); // changed their mind
      expect(await screen.findByText("bb")).toBeInTheDocument();
      releaseCh2();

      // later, opening chapter 2 starts it from the top, paused: no surprise seek-and-play
      await user.click(screen.getByTitle("next rendered chapter"));
      expect(await screen.findByText("sailed")).toBeInTheDocument();
      await waitFor(() => expect(screen.getByText("0:00 / 0:30")).toBeInTheDocument());
      expect(screen.getByText("paused")).toBeInTheDocument();
    });

    it("finishing the last recorded chapter forgets the place instead of offering its last second", async () => {
      const user = userEvent.setup();
      serveBook();
      localStorage.setItem("seiyuu.listen.b1", JSON.stringify({ chapter: 2, clip: "blk2:0", offset: 12, seconds: 12 }));
      const grabbed: { audio: HTMLAudioElement | null } = { audio: null };
      function Grab() {
        grabbed.audio = usePlayer()?.audio ?? null;
        return null;
      }
      renderWithProviders(
        <>
          <Listen />
          <TransportBar />
          <Grab />
        </>,
        { route: "/listen?book=b1" },
      );

      await user.click(await screen.findByRole("button", { name: "Resume chapter 2 at 0:12" }));
      expect(await screen.findByText("0:12 / 0:30")).toBeInTheDocument();
      await waitFor(() => expect(grabbed.audio).not.toBeNull());
      fireEvent(grabbed.audio!, new Event("ended")); // chapter 2 is the last recorded one
      await waitFor(() => expect(localStorage.getItem("seiyuu.listen.b1")).toBeNull());
    });
  });

  describe("the audiobook file", () => {
    it("offers Download .m4b when one exists", async () => {
      const server = serveBook();
      server.get("/api/books/b1", {
        ...detail,
        status: { ...detail.status, mastered: true },
        downloads: { m4b: { url: "/api/books/b1/files/m4b", bytes: 412 * 1024 * 1024 }, chapter_mp3s: [] },
      });
      renderListen();

      const link = await screen.findByRole("link", { name: /Download \.m4b/ });
      expect(link).toHaveAttribute("href", "/api/books/b1/files/m4b");
      expect(link).toHaveAttribute("download");
      expect(link).toHaveTextContent("412 MB");
    });

    it("without one, the Build the .m4b step sits there as a real link", async () => {
      serveBook();
      renderListen();

      expect(await screen.findByRole("link", { name: "Build the .m4b" })).toHaveAttribute(
        "href",
        "/render?book=b1#build",
      );
      expect(screen.queryByRole("link", { name: /Download \.m4b/ })).not.toBeInTheDocument();
    });

    it("audio recorded before the casting offers a real Record-again link", async () => {
      const server = serveBook();
      server.get("/api/books/b1", { ...detail, status: { ...detail.status, assigned: true } });
      server.get("/api/books/b1/render", { ...summary, mode: "single", active_mode: "single", available_modes: ["single"] });
      renderListen();

      expect(await screen.findByText("this audio predates your casting")).toBeInTheDocument();
      expect(screen.getByRole("link", { name: "Record again with your cast" })).toHaveAttribute(
        "href",
        "/render?book=b1#record",
      );
    });

    it("full-cast audio never shows the predates-casting nudge", async () => {
      const server = serveBook();
      server.get("/api/books/b1", { ...detail, status: { ...detail.status, assigned: true } });
      renderListen();

      await screen.findByRole("button", { name: "play/pause" });
      expect(screen.queryByText("this audio predates your casting")).not.toBeInTheDocument();
    });
  });
});
