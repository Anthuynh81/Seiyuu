import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { LexiconEntry, LexiconOut, SuggestedTerm } from "../api/types";
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

  it("shows the no-books empty state when nothing is ingested and never fetches a lexicon", async () => {
    const server = mockApi().get("/api/books", {
      books: [makeBook({ book_id: "b1", ingested: false })],
    });
    renderWithProviders(<Lexicon />);
    expect(await screen.findByText(/No ingested books yet/)).toBeInTheDocument();
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

    await user.click(screen.getByRole("button", { name: "+ add term" }));
    await user.type(screen.getAllByLabelText("term")[1], "kvothe");
    await user.type(screen.getAllByLabelText("respelling")[1], "kvo");

    expect(screen.getByText(/duplicate term: kvothe/)).toBeInTheDocument();
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

    await user.type(screen.getByLabelText("respelling"), "!");
    expect(previewBtn).toBeEnabled();
    await user.click(previewBtn);

    expect(await screen.findByText(/would re-synthesize 3 of 42 segments/)).toBeInTheDocument();
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
    await user.click(screen.getByRole("button", { name: "+ add term" }));
    await user.type(screen.getAllByLabelText("term")[1], " Denna ");
    await user.type(screen.getAllByLabelText("respelling")[1], " DEN-uh ");
    await user.click(screen.getAllByLabelText("case sensitive")[1]);
    // ...and an incomplete row (term without respelling) that must be dropped, not sent
    await user.click(screen.getByRole("button", { name: "+ add term" }));
    await user.type(screen.getAllByLabelText("term")[2], "Bast");

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
    await user.type(screen.getByLabelText("respelling"), "!");
    await user.click(screen.getByRole("button", { name: "save" }));

    // the message only renders once the editor is clean again, i.e. AFTER the refetch lands
    expect(
      await screen.findByText(/saved · 5 of 40 segments will re-synthesize on next render/),
    ).toBeInTheDocument();
    expect(
      server.calls.filter((c) => c.method === "GET" && c.url.includes("/lexicon")).length,
    ).toBeGreaterThan(1);

    // a real edit re-dirties the editor and retires the stale confirmation
    await user.type(screen.getByLabelText("respelling"), "?");
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
    expect(screen.getByText(/3 respellings from local\/\s*qwen3 — review and save/)).toBeInTheDocument();
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

    expect(await screen.findByText("anthropic respelling suggestions are a paid call")).toBeInTheDocument();
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
});
