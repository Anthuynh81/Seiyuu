import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { LexiconEntry, LexiconOut, SuggestedTerm, VoiceAssignment, VoiceOut } from "../api/types";
import { makeBook } from "../test/fixtures";
import { errorResponse, jsonResponse, mockApi, renderWithProviders } from "../test/utils";
import { Lexicon } from "./Lexicon";

function entry(overrides: Partial<LexiconEntry> & Pick<LexiconEntry, "term" | "respelling">): LexiconEntry {
  return { ipa: null, note: null, case_sensitive: false, ...overrides };
}

function lexiconOut(entries: LexiconEntry[], suggestions: SuggestedTerm[] = []): LexiconOut {
  return { book_id: "b1", schema_version: 1, entries, suggestions };
}

/** Mount the screen against one ingested book whose lexicon is `entries` (+ hard-name
    `suggestions`). Returns the mock server for per-test route additions and call asserts. */
function mountLexicon(entries: LexiconEntry[], suggestions: SuggestedTerm[] = []) {
  const server = mockApi()
    .get("/api/books", { books: [makeBook({ book_id: "b1", title: "The Name of the Wind" })] })
    .get("/api/books/b1/lexicon", lexiconOut(entries, suggestions));
  renderWithProviders(<Lexicon />);
  return server;
}

describe("Lexicon", () => {
  it("renders entries from GET /lexicon, offers only hard-name suggestions not already present, and a clicked chip seeds a new row", async () => {
    const user = userEvent.setup();
    mountLexicon(
      [entry({ term: "Kvothe", respelling: "KVOH-thee", ipa: "kvoʊθi" })],
      [
        { term: "Kvothe", count: 12, sample: "Kvothe smiled." },
        { term: "Elodin", count: 5, sample: "Elodin named the wind." },
      ],
    );

    expect(await screen.findByDisplayValue("Kvothe")).toBeInTheDocument();
    expect(screen.getByDisplayValue("KVOH-thee")).toBeInTheDocument();
    expect(screen.getByDisplayValue("kvoʊθi")).toBeInTheDocument();
    // the IPA column is explicitly marked as Kokoro-only in the entry editor
    expect(screen.getByPlaceholderText("IPA (Kokoro only)")).toBeInTheDocument();
    // Kvothe already has an entry, so only Elodin is suggested
    expect(screen.getByRole("button", { name: "Elodin ×5" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Kvothe ×12" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Elodin ×5" }));
    expect(screen.getByDisplayValue("Elodin")).toBeInTheDocument();
    // now that a row carries the term, the chip is filtered out
    expect(screen.queryByRole("button", { name: "Elodin ×5" })).not.toBeInTheDocument();
  });

  it("?term= (from Listen's Fix pronunciation) waits in a new unsaved row", async () => {
    const user = userEvent.setup();
    mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1", title: "The Name of the Wind" })] })
      .get("/api/books/b1/lexicon", lexiconOut([entry({ term: "Kvothe", respelling: "KVOH-thee" })]));
    renderWithProviders(<Lexicon />, { route: "/lexicon?book=b1&term=Elodin" });

    expect(await screen.findByDisplayValue("Elodin")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Kvothe")).toBeInTheDocument();
    // the word's row comes FIRST (easy to find in a long dictionary), its respelling focused
    const terms = screen.getAllByLabelText("word");
    expect(terms).toHaveLength(2);
    expect(terms[0]).toHaveValue("Elodin");
    const respelling = screen.getAllByLabelText("say it like")[0];
    await waitFor(() => expect(respelling).toHaveFocus());
    expect(screen.getByText(/Type how “Elodin” should sound/)).toBeInTheDocument();
    // nothing is saved for them: the row needs its respelling, then an explicit save
    expect(screen.getByRole("button", { name: "save" })).toBeDisabled();
    await user.type(respelling, "EL-oh-din");
    expect(screen.getByRole("button", { name: "save" })).toBeEnabled();
    // the hint only asks while the row is still waiting for its respelling
    expect(screen.queryByText(/Type how “Elodin” should sound/)).not.toBeInTheDocument();
  });

  it("?term= for a word already in the dictionary adds no duplicate row", async () => {
    mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1" })] })
      .get("/api/books/b1/lexicon", lexiconOut([entry({ term: "Kvothe", respelling: "KVOH-thee" })]));
    renderWithProviders(<Lexicon />, { route: "/lexicon?book=b1&term=kvothe" });

    expect(await screen.findByDisplayValue("Kvothe")).toBeInTheDocument();
    expect(screen.getAllByLabelText("word")).toHaveLength(1);
    // already respelled: nothing left to ask for
    expect(screen.queryByText(/should sound/)).not.toBeInTheDocument();
  });

  it("shows the no-books empty state when nothing is ingested and never fetches a lexicon", async () => {
    const server = mockApi().get("/api/books", {
      books: [makeBook({ book_id: "b1", ingested: false })],
    });
    renderWithProviders(<Lexicon />);
    expect(await screen.findByText(/No imported books yet/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Add a book" })).toHaveAttribute("href", "/#add");
    expect(server.lastCall("GET", "/lexicon")).toBeUndefined();
  });

  it("surfaces the API error message when the lexicon fails to load", async () => {
    mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1" })] })
      .error("GET", "/api/books/b1/lexicon", 500, "lexicon_unreadable", "lexicon file corrupted");
    renderWithProviders(<Lexicon />);
    expect(await screen.findByText("lexicon file corrupted")).toBeInTheDocument();
  });

  it("flags a case-insensitive duplicate term and disables save and preview", async () => {
    const user = userEvent.setup();
    mountLexicon([entry({ term: "Kvothe", respelling: "KVOH-thee" })]);
    await screen.findByDisplayValue("Kvothe");

    await user.click(screen.getByRole("button", { name: "+ add a word" }));
    await user.type(screen.getAllByLabelText("word")[1], "kvothe");
    await user.type(screen.getAllByLabelText("say it like")[1], "kvo");

    expect(screen.getByText(/listed twice: kvothe/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "save" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "preview impact" })).toBeDisabled();
  });

  it("preview impact is disabled until dirty, then POSTs the cleaned proposal and shows the affected-segment count before commit", async () => {
    const user = userEvent.setup();
    const server = mountLexicon([entry({ term: "Kvothe", respelling: "KVOH-thee", ipa: "kvoʊθi" })]);
    server.post("/api/books/b1/lexicon/preview", { affected_blocks: 3, total_speakable_blocks: 42 });

    await screen.findByDisplayValue("KVOH-thee");
    const previewBtn = screen.getByRole("button", { name: "preview impact" });
    expect(previewBtn).toBeDisabled(); // editor matches the server copy

    await user.type(screen.getByLabelText("say it like"), "!");
    expect(previewBtn).toBeEnabled();
    await user.click(previewBtn);

    expect(await screen.findByText(/would re-record 3 of 42 paragraphs/)).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/lexicon/preview")).toEqual({
      entries: [{ term: "Kvothe", respelling: "KVOH-thee!", ipa: "kvoʊθi", note: null, case_sensitive: false }],
    });
  });

  it("save PUTs the full cleaned entries array: trimmed fields, blank optionals nulled, incomplete rows dropped", async () => {
    const user = userEvent.setup();
    const server = mountLexicon([entry({ term: "Kvothe", respelling: "KVOH-thee", ipa: "kvoʊθi" })]);
    server.put("/api/books/b1/lexicon", {
      book_id: "b1",
      schema_version: 1,
      entries: [],
      affected_blocks: 5,
      total_speakable_blocks: 40,
    });

    await screen.findByDisplayValue("KVOH-thee");
    const saveBtn = screen.getByRole("button", { name: "save" });
    expect(saveBtn).toBeDisabled(); // clean editor

    // a complete row with padding to trim, marked case-sensitive...
    await user.click(screen.getByRole("button", { name: "+ add a word" }));
    await user.type(screen.getAllByLabelText("word")[1], " Denna ");
    await user.type(screen.getAllByLabelText("say it like")[1], " DEN-uh ");
    await user.click(screen.getAllByLabelText("match capitals exactly")[1]);
    // ...and an incomplete row (term without respelling) that must be dropped, not sent
    await user.click(screen.getByRole("button", { name: "+ add a word" }));
    await user.type(screen.getAllByLabelText("word")[2], "Bast");

    expect(saveBtn).toBeEnabled();
    await user.click(saveBtn);

    await waitFor(() => expect(server.lastCall("PUT", "/api/books/b1/lexicon")).toBeDefined());
    expect(server.jsonBodyOf("PUT", "/api/books/b1/lexicon")).toEqual({
      entries: [
        { term: "Kvothe", respelling: "KVOH-thee", ipa: "kvoʊθi", note: null, case_sensitive: false },
        { term: "Denna", respelling: "DEN-uh", ipa: null, note: null, case_sensitive: true },
      ],
    });
  });

  it("the saved confirmation survives the lexicon refetch that lands its own save, and retires on the next edit", async () => {
    const user = userEvent.setup();
    // the GET route serves a mutable copy so the post-save invalidation refetch returns
    // exactly what the PUT stored — the round-trip that used to clear the message
    let serverEntries = [entry({ term: "Kvothe", respelling: "KVOH-thee" })];
    const server = mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1", title: "The Name of the Wind" })] })
      .on("GET", "/api/books/b1/lexicon", () => jsonResponse(lexiconOut(serverEntries)));
    server.on("PUT", "/api/books/b1/lexicon", (_url, init) => {
      serverEntries = (JSON.parse(init?.body as string) as { entries: LexiconEntry[] }).entries;
      return jsonResponse({
        book_id: "b1",
        schema_version: 1,
        entries: serverEntries,
        affected_blocks: 5,
        total_speakable_blocks: 40,
      });
    });
    renderWithProviders(<Lexicon />);

    await screen.findByDisplayValue("KVOH-thee");
    await user.type(screen.getByLabelText("say it like"), "!");
    await user.click(screen.getByRole("button", { name: "save" }));

    // the message only renders once the editor is clean again, i.e. AFTER the refetch lands
    expect(
      await screen.findByText(/saved · 5 of 40 paragraphs will be re-recorded the next time you record/),
    ).toBeInTheDocument();
    expect(
      server.calls.filter((c) => c.method === "GET" && c.url.includes("/lexicon")).length,
    ).toBeGreaterThan(1);

    // a real edit re-dirties the editor and retires the stale confirmation
    await user.type(screen.getByLabelText("say it like"), "?");
    expect(screen.queryByText(/saved ·/)).not.toBeInTheDocument();
  });

  it("suggest-with-AI POSTs the editor's terms and folds advisory respellings in without clobbering user input", async () => {
    const user = userEvent.setup();
    const server = mountLexicon(
      [entry({ term: "Auri", respelling: "OW-ree" })],
      [{ term: "Kvothe", count: 12, sample: "Kvothe smiled." }],
    );
    server.post("/api/books/b1/lexicon/suggest-respellings", {
      provider: "local",
      model: "qwen3",
      suggestions: [
        { term: "Kvothe", respelling: "KVOH-thee", note: "rhymes with quoth" },
        { term: "Auri", respelling: "AH-ree", note: null },
        { term: "Denna", respelling: "DEN-uh", note: null },
      ],
    });

    await screen.findByDisplayValue("OW-ree");
    await user.click(screen.getByRole("button", { name: "Kvothe ×12" })); // row with an empty respelling
    await user.click(screen.getByRole("button", { name: /suggest with AI/ }));

    // fills the empty Kvothe respelling + note, appends the unknown Denna as a new row,
    // and leaves the user's own OW-ree untouched
    expect(await screen.findByDisplayValue("KVOH-thee")).toBeInTheDocument();
    expect(screen.getByDisplayValue("rhymes with quoth")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Denna")).toBeInTheDocument();
    expect(screen.getByDisplayValue("DEN-uh")).toBeInTheDocument();
    expect(screen.getByDisplayValue("OW-ree")).toBeInTheDocument();
    expect(screen.queryByDisplayValue("AH-ree")).not.toBeInTheDocument();
    expect(screen.getByText(/3 suggestions from the AI — review and save/)).toBeInTheDocument();
    // which model answered is a tooltip detail, never the main text
    expect(screen.queryByText(/qwen3/)).not.toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "suggest-respellings")).toEqual({
      terms: ["Auri", "Kvothe"],
      confirm_paid: false,
    });
  });

  it("a 402 from the paid suggester gates retry behind explicit approval, then re-sends confirm_paid=true", async () => {
    const user = userEvent.setup();
    const server = mountLexicon([entry({ term: "Auri", respelling: "OW-ree" })]);
    server.on("POST", "/api/books/b1/lexicon/suggest-respellings", (_url, init) => {
      const body = JSON.parse(String(init?.body)) as { confirm_paid?: boolean };
      return body.confirm_paid
        ? jsonResponse({
            provider: "anthropic",
            model: "claude-sonnet",
            suggestions: [{ term: "Denna", respelling: "DEN-uh", note: null }],
          })
        : errorResponse(402, "payment_confirmation_required", "anthropic respelling suggestions are a paid call");
    });

    await screen.findByDisplayValue("OW-ree");
    await user.click(screen.getByRole("button", { name: /suggest with AI/ }));

    // plain words, not the server's message
    expect(await screen.findByText(/This uses a paid online service.*Nothing has been spent yet/)).toBeInTheDocument();
    expect(screen.queryByText("anthropic respelling suggestions are a paid call")).not.toBeInTheDocument();
    const retry = screen.getByRole("button", { name: "retry with AI" });
    expect(retry).toBeDisabled(); // never re-send without the explicit approval

    await user.click(screen.getByRole("checkbox", { name: "approve the paid (Anthropic) suggester" }));
    expect(retry).toBeEnabled();
    await user.click(retry);

    expect(await screen.findByDisplayValue("DEN-uh")).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "suggest-respellings")).toEqual({ terms: ["Auri"], confirm_paid: true });
    // success clears the paid-confirm gate
    expect(screen.queryByRole("checkbox", { name: /approve the paid/ })).not.toBeInTheDocument();
  });

  it("?book=&term= deep link opens THAT book (not the first) with the word waiting", async () => {
    const server = mockApi()
      .get("/api/books", {
        books: [makeBook({ book_id: "b1", title: "First" }), makeBook({ book_id: "b2", title: "Second" })],
      })
      .get("/api/books/b2/lexicon", lexiconOut([]));
    renderWithProviders(<Lexicon />, { route: "/lexicon?book=b2&term=Tarbean" });

    expect(await screen.findByDisplayValue("Tarbean")).toBeInTheDocument();
    expect(server.lastCall("GET", "/api/books/b1/lexicon")).toBeUndefined();
    expect(screen.getByText(/Type how “Tarbean” should sound/)).toBeInTheDocument();
  });

  it("speaks plain words: words, lines/paragraphs, match capitals exactly — no segments", async () => {
    mountLexicon([entry({ term: "Kvothe", respelling: "KVOH-thee" })]);
    await screen.findByDisplayValue("Kvothe");
    expect(screen.getByRole("columnheader", { name: "match capitals exactly" })).toBeInTheDocument();
    expect(screen.getByLabelText("match capitals exactly")).toBeInTheDocument();
    expect(screen.queryByText(/segment/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/re-synthes/i)).not.toBeInTheDocument();
    expect(screen.getByText(/re-records only the\s+lines/)).toBeInTheDocument();
  });

  it("IPA lives under Advanced: hidden until asked for, shown when an entry already uses it", async () => {
    const user = userEvent.setup();
    mountLexicon([entry({ term: "Kvothe", respelling: "KVOH-thee" })]);
    await screen.findByDisplayValue("Kvothe");
    expect(screen.queryByLabelText("phonetic spelling (IPA)")).not.toBeInTheDocument();

    const toggle = screen.getByRole("button", { name: "advanced: IPA" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await user.click(toggle);
    expect(screen.getByLabelText("phonetic spelling (IPA)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "hide advanced" })).toHaveAttribute("aria-expanded", "true");
  });

  it("after a save that changes lines of a recorded book, links to Produce to re-record them", async () => {
    const user = userEvent.setup();
    let serverEntries = [entry({ term: "Kvothe", respelling: "KVOH-thee" })];
    const server = mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1", rendered: true })] })
      .on("GET", "/api/books/b1/lexicon", () => jsonResponse(lexiconOut(serverEntries)));
    server.on("PUT", "/api/books/b1/lexicon", (_url, init) => {
      serverEntries = (JSON.parse(init?.body as string) as { entries: LexiconEntry[] }).entries;
      return jsonResponse({
        book_id: "b1",
        schema_version: 1,
        entries: serverEntries,
        affected_blocks: 2,
        total_speakable_blocks: 40,
      });
    });
    renderWithProviders(<Lexicon />);

    await screen.findByDisplayValue("KVOH-thee");
    await user.type(screen.getByLabelText("say it like"), "e");
    await user.click(screen.getByRole("button", { name: "save" }));
    const link = await screen.findByRole("link", { name: /re-record the changed chapters on Produce/ });
    expect(link).toHaveAttribute("href", "/render?book=b1");
  });

  it("a refused save reads in plain words, the server's text only under Details, and is not re-sent", async () => {
    const user = userEvent.setup();
    const server = mountLexicon([entry({ term: "Kvothe", respelling: "KVOH-thee" })]);
    // put_lexicon's real refusal when the saved dictionary can't be read (_load_lexicon_or_500)
    server.error("PUT", "/api/books/b1/lexicon", 500, "corrupt_artifact", "lexicon.json for 'b1' is unreadable: bad json");

    await screen.findByDisplayValue("KVOH-thee");
    await user.type(screen.getByLabelText("say it like"), "e");
    await user.click(screen.getByRole("button", { name: "save" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("One of this book's saved files is damaged and couldn't be read.");
    expect(within(alert).getByText("Details")).toBeInTheDocument();
    expect(alert.querySelector(".joberr-sum")).not.toHaveTextContent(/lexicon\.json|corrupt_artifact/);
    expect(server.calls.filter((c) => c.method === "PUT")).toHaveLength(1);
  });

  it("the Advanced choice belongs to one book: switching to a book with IPA shows it again", async () => {
    const user = userEvent.setup();
    mockApi()
      .get("/api/books", {
        books: [makeBook({ book_id: "b1", title: "First" }), makeBook({ book_id: "b2", title: "Second" })],
      })
      .get("/api/books/b1/lexicon", lexiconOut([entry({ term: "Kvothe", respelling: "KVOH-thee", ipa: "kvoʊθi" })]))
      .get("/api/books/b2/lexicon", {
        ...lexiconOut([entry({ term: "Denna", respelling: "DEN-uh", ipa: "dɛnə" })]),
        book_id: "b2",
      });
    renderWithProviders(<Lexicon />, { route: "/lexicon?book=b1" });

    expect(await screen.findByDisplayValue("kvoʊθi")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "hide advanced" }));
    expect(screen.queryByDisplayValue("kvoʊθi")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^First/ }));
    await user.click(await screen.findByRole("option", { name: /Second/ }));
    expect(await screen.findByDisplayValue("dɛnə")).toBeInTheDocument();
  });

  describe("hear it in a sentence", () => {
    const narrator: VoiceOut = {
      voice_id: "v-narr",
      name: "Narrator",
      kind: "preset",
      engine: "kokoro",
      preset_id: "af_heart",
      blend: null,
      reference_audio: null,
      seed: 1,
      consent_attested: false,
      consent: null,
      tags: [],
      created_at: "2026-07-11T00:00:00Z",
      has_audition: false,
    };
    const assignment: VoiceAssignment = {
      schema_version: 1,
      book_id: "b1",
      stage: "draft",
      narrator_voice_id: "v-narr",
      assignments: {},
      thought_voice_id: null,
      created_at: "2026-07-11T00:00:00Z",
    };
    const castBook = (lex: LexiconOut) =>
      mockApi()
        .get("/api/books", { books: [makeBook({ book_id: "b1", attributed: true, assigned: true })] })
        .get("/api/books/b1/lexicon", lex)
        .get("/api/books/b1/assignment", assignment)
        .get("/api/voices", { voices: [narrator] });

    it("a new row from a hard-name chip plays the book's own sentence with the respelling swapped in", async () => {
      const user = userEvent.setup();
      // the backend leaves saved terms out of the suggestions (suggest_terms existing_terms=),
      // so only a row added from a chip has the book's sentence to hand
      const server = castBook(
        lexiconOut([], [{ term: "Kvothe", count: 12, sample: "Then Kvothe smiled at kvothe." }]),
      ).post("/api/voices/v-narr/audition", {
        voice_id: "v-narr",
        duration_seconds: 1.2,
        cost_usd: 0,
        audition_url: "/api/voices/v-narr/audition.wav",
      });
      renderWithProviders(<Lexicon />);

      await user.click(await screen.findByRole("button", { name: "Kvothe ×12" }));
      await user.type(screen.getByLabelText("say it like"), "KVOH-thee");
      await user.click(screen.getByRole("button", { name: "hear “Kvothe” in a sentence" }));
      await waitFor(() => expect(server.lastCall("POST", "/api/voices/v-narr/audition")).toBeDefined());
      expect(server.jsonBodyOf("POST", "/api/voices/v-narr/audition")).toEqual({
        text: "Then KVOH-thee smiled at KVOH-thee.",
        confirm_paid: false,
      });
      // labelled approximate
      expect(screen.getByText(/approximate/)).toBeInTheDocument();
    });

    // a SAVED entry has no book sentence (see above): it plays the short fixed one
    it("a GPU refusal is shown plainly and never retried on its own", async () => {
      const user = userEvent.setup();
      const server = castBook(lexiconOut([entry({ term: "Kvothe", respelling: "KVOH-thee" })])).error(
        "POST",
        "/api/voices/v-narr/audition",
        409,
        "gpu_busy",
        "a render job holds the GPU",
        { kind: "render" },
      );
      renderWithProviders(<Lexicon />);

      await user.click(await screen.findByRole("button", { name: "hear “Kvothe” in a sentence" }));
      // the notice row (and the row's live region announcing it)
      expect((await screen.findAllByText(/Busy recording — try after it finishes/)).length).toBeGreaterThan(0);
      expect(screen.getByRole("button", { name: "try again" })).toBeInTheDocument();
      expect(server.jsonBodyOf("POST", "/api/voices/v-narr/audition")).toEqual({
        text: "This is how you say KVOH-thee.",
        confirm_paid: false,
      });
      await new Promise((r) => setTimeout(r, 50));
      expect(server.calls.filter((c) => c.method === "POST")).toHaveLength(1);
    });

    it("an uncast book greys the key with the reason, and never asks for an assignment", async () => {
      const server = mountLexicon([entry({ term: "Kvothe", respelling: "KVOH-thee" })]);
      const key = await screen.findByRole("button", { name: /hear “Kvothe” in a sentence — cast the voices first/ });
      expect(key).toHaveAttribute("aria-disabled", "true");
      expect(server.lastCall("GET", "/assignment")).toBeUndefined();
    });
  });
});
