import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { AssignmentDraftResponse, LinkSuggestion, Series as SeriesModel } from "../api/types";
import { makeBook } from "../test/fixtures";
import { mockApi, renderWithProviders } from "../test/utils";
import { Series } from "./Series";

/* F5 precision rules under test: suggestions are confirm-then-apply (never auto-applied),
   save-cast is the only write-back (nothing learned silently), and a deleted linked voice
   degrades instead of crashing or being inherited. */

function castBook() {
  return makeBook({ book_id: "b1", title: "The Final Empire", attributed: true, assigned: true });
}

function makeSeries(overrides: Partial<SeriesModel> = {}): SeriesModel {
  return {
    series_id: "s1",
    name: "Mistborn",
    book_ids: ["b1"],
    voice_links: { vin: "voice_vin" },
    ...overrides,
  };
}

function makeSuggestion(overrides: Partial<LinkSuggestion> = {}): LinkSuggestion {
  return {
    character_id: "c_vin",
    canonical_name: "Vin",
    identity_key: "vin",
    voice_id: "voice_vin",
    voice_exists: true,
    ...overrides,
  };
}

const SUGGESTIONS_URL = "/api/series/s1/books/b1/link-suggestions";

function draftResponse(): AssignmentDraftResponse {
  return {
    assignment: {
      schema_version: 1,
      book_id: "b1",
      stage: "draft",
      narrator_voice_id: "voice_narr",
      assignments: { c_vin: "voice_vin" },
      thought_voice_id: null,
      created_at: "2026-07-11T00:00:00Z",
    },
    created_voice_ids: [],
    edit_warnings: [],
  };
}

describe("Series", () => {
  it("renders the selected series from /api/series — membership, voice links, unlink actions", async () => {
    mockApi()
      .get("/api/books", { books: [castBook()] })
      .get("/api/series", {
        series: [
          makeSeries({
            book_ids: ["b1", "b0"],
            voice_links: { vin: "voice_vin", "lord ruler": "voice_lr" },
          }),
        ],
      })
      .get(SUGGESTIONS_URL, { series_id: "s1", book_id: "b1", suggestions: [] });
    renderWithProviders(<Series />, { route: "/series?book=b1&series=s1" });

    expect(await screen.findByText("2 book(s)")).toBeInTheDocument();
    expect(screen.getByText("2 linked voice(s)")).toBeInTheDocument();
    expect(screen.getByText("1 series in the library")).toBeInTheDocument();
    // membership chips: the current book is marked, the sibling falls back to its id
    expect(screen.getByText("· current")).toBeInTheDocument();
    expect(screen.getByText("b0")).toBeInTheDocument();
    // the learned cast, one unlink action per link
    expect(screen.getByText("voice_vin")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "unlink vin" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "unlink lord ruler" })).toBeInTheDocument();
  });

  it("creating a series POSTs {name, book_id} and clears the field on success", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/books", { books: [castBook()] })
      .get("/api/series", { series: [] })
      .post("/api/series", makeSeries());
    renderWithProviders(<Series />, { route: "/series?book=b1" });

    const nameInput = await screen.findByRole("textbox", { name: "series name" });
    const createBtn = screen.getByRole("button", { name: "create series" });
    expect(createBtn).toBeDisabled(); // an empty name can't create
    await user.type(nameInput, "Mistborn");
    await user.click(createBtn);

    await waitFor(() => expect(nameInput).toHaveValue(""));
    expect(server.jsonBodyOf("POST", "/api/series")).toEqual({ name: "Mistborn", book_id: "b1" });
  });

  it("surfaces the backend rejection as readable text when creating a series fails", async () => {
    const user = userEvent.setup();
    mockApi()
      .get("/api/books", { books: [castBook()] })
      .get("/api/series", { series: [] })
      .error("POST", "/api/series", 409, "book_not_assigned", "book b1 has no cast — assign voices first");
    renderWithProviders(<Series />, { route: "/series?book=b1" });

    await user.type(await screen.findByRole("textbox", { name: "series name" }), "Mistborn");
    await user.click(screen.getByRole("button", { name: "create series" }));

    expect(await screen.findByText("book b1 has no cast — assign voices first")).toBeInTheDocument();
  });

  it("an unattributed, uncast book gates create and save-cast and never fetches suggestions", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/books", {
        books: [makeBook({ book_id: "b1", title: "Fresh", attributed: false, assigned: false })],
      })
      .get("/api/series", { series: [makeSeries()] });
    renderWithProviders(<Series />, { route: "/series?book=b1&series=s1" });

    // stage prerequisite refusal instead of a suggestions fetch
    expect(await screen.findByText(/this book has no attribution yet/)).toBeInTheDocument();
    expect(server.lastCall("GET", "link-suggestions")).toBeUndefined();

    expect(screen.getByRole("button", { name: "save cast to series" })).toBeDisabled();
    await user.type(screen.getByRole("textbox", { name: "series name" }), "Wax and Wayne");
    expect(screen.getByRole("button", { name: "create series" })).toBeDisabled();
  });

  it("attaching a sibling book POSTs membership only, then the panel flips to member actions", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/books", { books: [castBook()] })
      .get("/api/series", { series: [makeSeries({ book_ids: ["b0"] })] })
      .get(SUGGESTIONS_URL, { series_id: "s1", book_id: "b1", suggestions: [] })
      .post("/api/series/s1/books", makeSeries({ book_ids: ["b0", "b1"] }));
    renderWithProviders(<Series />, { route: "/series?book=b1&series=s1" });

    const addBtn = await screen.findByRole("button", { name: "add this book to the series" });
    // the post-mutation refetch sees the updated membership
    server.get("/api/series", { series: [makeSeries({ book_ids: ["b0", "b1"] })] });
    await user.click(addBtn);

    expect(await screen.findByRole("button", { name: "save cast to series" })).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/series/s1/books")).toEqual({ book_id: "b1" });
  });

  it("suggestions render for confirmation — a deleted voice degrades — and apply sends only confirmed overrides", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/books", { books: [castBook()] })
      .get("/api/series", {
        series: [makeSeries({ voice_links: { vin: "voice_vin", kelsier: "voice_kel" } })],
      })
      .get(SUGGESTIONS_URL, {
        series_id: "s1",
        book_id: "b1",
        suggestions: [
          makeSuggestion(),
          makeSuggestion({
            character_id: "c_kel",
            canonical_name: "Kelsier",
            identity_key: "kelsier",
            voice_id: "voice_kel",
            voice_exists: false,
          }),
        ],
      })
      .post("/api/books/b1/assignment/draft", draftResponse());
    renderWithProviders(<Series />, { route: "/series?book=b1&series=s1" });

    // the intact link defaults confirmed; the deleted one renders disabled + labeled, not inherited
    const vin = await screen.findByRole("checkbox", { name: "inherit Vin" });
    expect(vin).toBeChecked();
    const kel = screen.getByRole("checkbox", { name: "inherit Kelsier" });
    expect(kel).toBeDisabled();
    expect(kel).not.toBeChecked();
    expect(screen.getByText("voice deleted")).toBeInTheDocument();

    // nothing was applied before the explicit click
    expect(server.lastCall("POST", "/assignment/draft")).toBeUndefined();
    await user.click(screen.getByRole("button", { name: "apply 1 inherited voice(s)" }));

    expect(await screen.findByText(/applied · 1 inherited/)).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
      strategy: "smart",
      overrides: { c_vin: "voice_vin" },
    });
  });

  it("unconfirming a suggestion excludes it from the applied overrides", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/books", { books: [castBook()] })
      .get("/api/series", { series: [makeSeries()] })
      .get(SUGGESTIONS_URL, {
        series_id: "s1",
        book_id: "b1",
        suggestions: [
          makeSuggestion(),
          makeSuggestion({
            character_id: "c_saz",
            canonical_name: "Sazed",
            identity_key: "sazed",
            voice_id: "voice_saz",
          }),
        ],
      })
      .post("/api/books/b1/assignment/draft", draftResponse());
    renderWithProviders(<Series />, { route: "/series?book=b1&series=s1" });

    await user.click(await screen.findByRole("checkbox", { name: "inherit Vin" }));
    await user.click(screen.getByRole("button", { name: "apply 1 inherited voice(s)" }));

    await waitFor(() =>
      expect(server.jsonBodyOf("POST", "/api/books/b1/assignment/draft")).toEqual({
        strategy: "smart",
        overrides: { c_saz: "voice_saz" },
      }),
    );
  });

  it("save-cast is an explicit POST to /save-cast and reports the links written back", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/books", { books: [castBook()] })
      .get("/api/series", { series: [makeSeries()] })
      .get(SUGGESTIONS_URL, { series_id: "s1", book_id: "b1", suggestions: [] })
      .post("/api/series/s1/save-cast", {
        series: makeSeries({ voice_links: { vin: "voice_vin", elend: "voice_el" } }),
        linked_keys: ["vin", "elend"],
      });
    renderWithProviders(<Series />, { route: "/series?book=b1&series=s1" });

    const saveBtn = await screen.findByRole("button", { name: "save cast to series" });
    expect(server.lastCall("POST", "save-cast")).toBeUndefined(); // nothing learned silently
    await user.click(saveBtn);

    expect(await screen.findByText("saved · 2 link(s) added / updated")).toBeInTheDocument();
    expect(server.jsonBodyOf("POST", "/api/series/s1/save-cast")).toEqual({ book_id: "b1" });
  });

  it("unlink DELETEs /links with the identity key url-encoded and the row drops on refetch", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/books", { books: [castBook()] })
      .get("/api/series", { series: [makeSeries({ voice_links: { "lord ruler": "voice_lr" } })] })
      .get(SUGGESTIONS_URL, { series_id: "s1", book_id: "b1", suggestions: [] })
      .delete("/api/series/s1/links?name=lord%20ruler", makeSeries({ voice_links: {} }));
    renderWithProviders(<Series />, { route: "/series?book=b1&series=s1" });

    const unlinkBtn = await screen.findByRole("button", { name: "unlink lord ruler" });
    server.get("/api/series", { series: [makeSeries({ voice_links: {} })] });
    await user.click(unlinkBtn);

    expect(await screen.findByText(/no voice links yet/)).toBeInTheDocument();
    expect(server.lastCall("DELETE", "/api/series/s1/links")?.url).toBe(
      "/api/series/s1/links?name=lord%20ruler",
    );
  });
});
