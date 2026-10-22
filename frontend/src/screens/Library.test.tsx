import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { makeBook, makeEngine, makeJob, makeSystemProbe } from "../test/fixtures";
import { jsonResponse, mockApi, renderWithProviders } from "../test/utils";
import { Library } from "./Library";

describe("Library", () => {
  it("renders the shelf from /api/books with stage-appropriate actions", async () => {
    mockApi().get("/api/books", {
      books: [
        makeBook({ book_id: "b1", title: "Rendered One", rendered: true, attributed: true, assigned: true }),
        makeBook({ book_id: "b2", title: "Fresh Ingest" }),
      ],
    });
    renderWithProviders(<Library />);

    expect(await screen.findByText("Rendered One")).toBeInTheDocument();
    expect(screen.getByText("Fresh Ingest")).toBeInTheDocument();
    // rendered+attributed book gets listen + review; the fresh one only render & jobs
    expect(screen.getByRole("button", { name: "▶ listen" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "review characters" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "render & jobs" })).toHaveLength(2);
  });

  describe("next step on each card", () => {
    it("is one primary link per card, in plain words, to where that step happens", async () => {
      mockApi().get("/api/books", {
        books: [
          makeBook({ book_id: "b1", title: "Rendered One", rendered: true, attributed: true, assigned: true }),
          makeBook({ book_id: "b2", title: "Fresh Ingest" }),
          makeBook({ book_id: "b3", title: "Has Speakers", attributed: true }),
        ],
      });
      const { container } = renderWithProviders(<Library />);
      await screen.findByText("Rendered One");

      const card = (title: string) => screen.getByText(title).closest<HTMLElement>(".bookcard")!;
      expect(within(card("Fresh Ingest")).getByRole("link", { name: "Find the speakers" })).toHaveAttribute(
        "href",
        "/render?book=b2#find-speakers",
      );
      expect(within(card("Has Speakers")).getByRole("link", { name: "Cast the voices" })).toHaveAttribute(
        "href",
        "/review?book=b3",
      );
      expect(within(card("Rendered One")).getByRole("link", { name: "Build the .m4b" })).toHaveAttribute(
        "href",
        "/render?book=b1#build",
      );
      // exactly one primary per card; the old actions stay, but quiet
      expect(container.querySelectorAll(".bookcard .nextrow .key")).toHaveLength(3);
      expect(screen.getByRole("button", { name: "▶ listen" })).toHaveClass("quiet");
    });

    it("a finished book's next step is Listen, and the duplicate listen key goes away", async () => {
      mockApi().get("/api/books", {
        books: [
          makeBook({
            book_id: "b1",
            title: "Done",
            attributed: true,
            assigned: true,
            rendered: true,
            assembled: true,
            mastered: true,
          }),
        ],
      });
      renderWithProviders(<Library />);
      expect(await screen.findByRole("link", { name: "Listen" })).toHaveAttribute("href", "/listen?book=b1");
      expect(screen.queryByRole("button", { name: "▶ listen" })).not.toBeInTheDocument();
    });

    it("while a job runs the step is disabled and says what is happening", async () => {
      mockApi().get("/api/books", {
        books: [makeBook({ book_id: "b1", active_job: { job_id: "j1", kind: "attribute", state: "running" } })],
      });
      renderWithProviders(<Library />);
      const busy = await screen.findByRole("button", { name: "Finding the speakers…" });
      expect(busy).toBeDisabled();
      expect(busy).toHaveAccessibleDescription(/bar at the bottom/);
      expect(screen.queryByRole("link", { name: "Find the speakers" })).not.toBeInTheDocument();
    });

    it("after a failure the step is a retry, and the banner alone carries the fix", async () => {
      mockApi()
        .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Failed Once", attributed: true, assigned: true })] })
        .get("/api/jobs?limit=200", {
          jobs: [
            makeJob({
              book_id: "b1",
              kind: "render",
              state: "failed",
              is_terminal: true,
              error: "GpuBusyError: another seiyuu process holds the GPU",
            }),
          ],
        });
      renderWithProviders(<Library />);
      const retry = await screen.findByRole("link", { name: "Record again" });
      expect(retry).toHaveAttribute("href", "/render?book=b1#record");
      // the fix is said once (in the banner), not repeated beside the button
      expect(screen.getAllByText(/Close it or wait for it to finish/)).toHaveLength(1);
      expect(retry).not.toHaveAccessibleDescription(/Close it or wait/);
    });

    it("after the banner is dismissed, the retry carries the fix itself (also after a reload)", async () => {
      const user = userEvent.setup();
      mockApi()
        .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Failed Once", attributed: true, assigned: true })] })
        .get("/api/jobs?limit=200", {
          jobs: [
            makeJob({
              job_id: "f-9",
              book_id: "b1",
              kind: "render",
              state: "failed",
              is_terminal: true,
              error: "GpuBusyError: another seiyuu process holds the GPU",
            }),
          ],
        });
      const first = renderWithProviders(<Library />);
      const banner = await screen.findByRole("group", { name: "The last recording failed" });
      await user.click(within(banner).getByRole("button", { name: "Dismiss" }));
      const retry = screen.getByRole("link", { name: "Record again" });
      expect(retry).toHaveAccessibleDescription(/Close it or wait for it to finish/);
      expect(screen.getAllByText(/Close it or wait for it to finish/)).toHaveLength(1);

      first.unmount();
      renderWithProviders(<Library />);
      expect(await screen.findByRole("link", { name: "Record again" })).toHaveAccessibleDescription(
        /Close it or wait for it to finish/,
      );
    });

    it("a failure for another step leaves the banner and the step's own reason both showing", async () => {
      mockApi()
        .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Unattributed" })] })
        .get("/api/jobs?limit=200", {
          jobs: [makeJob({ book_id: "b1", kind: "render", state: "failed", is_terminal: true, error: "boom" })],
        });
      renderWithProviders(<Library />);
      expect(await screen.findByRole("group", { name: "The last recording failed" })).toBeInTheDocument();
      expect(screen.getByRole("link", { name: "Find the speakers" })).toHaveAccessibleDescription(
        /who says each line/,
      );
    });

    it("'Add the book again' lands on the upload slot and focuses it", async () => {
      mockApi().get("/api/books", { books: [makeBook({ book_id: "b1", title: "No Text", ingested: false })] });
      const user = userEvent.setup();
      renderWithProviders(<Library />);
      const again = await screen.findByRole("link", { name: "Add the book again" });
      expect(again).toHaveAttribute("href", "/#add");
      await user.click(again);
      await waitFor(() => expect(document.activeElement).toHaveAttribute("id", "add"));
      expect(document.activeElement).toHaveTextContent(/drop epub or pdf/);
    });
  });

  it("surfaces the API error message when the shelf fails to load", async () => {
    mockApi().error("GET", "/api/books", 500, "boom", "the shelf collapsed");
    renderWithProviders(<Library />);
    expect(await screen.findByText("the shelf collapsed")).toBeInTheDocument();
  });

  it("delete escalates through the 402 paid-artifacts confirm and re-sends confirm_paid=true", async () => {
    const user = userEvent.setup();
    const server = mockApi().get("/api/books", { books: [makeBook({ book_id: "b1", title: "Costly" })] });
    server.on("DELETE", /\/api\/books\/b1\?confirm_paid=false/, () =>
      jsonResponse(
        {
          error: {
            code: "payment_confirmation_required",
            message: "paid renders exist",
            detail: { paid_segment_count: 7, estimated_usd: 1.25, engines: ["elevenlabs"], paid_voice_ids: ["v9"] },
          },
        },
        402,
      ),
    );
    server.delete(/\/api\/books\/b1\?confirm_paid=true/, { deleted: "b1" });
    renderWithProviders(<Library />);

    await user.click(await screen.findByRole("button", { name: "delete" }));
    await user.click(screen.getByRole("button", { name: "delete book" }));

    // the 402 escalates: the dialog re-titles and spells out the paid segments
    expect(await screen.findByRole("dialog", { name: "Discard paid renders?" })).toBeInTheDocument();
    const discard = screen.getByRole("button", { name: /discard 7 paid segment\(s\) & delete/ });
    await user.click(discard);

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.lastCall("DELETE", "confirm_paid=true")).toBeDefined();
  });

  it("a conflicting_job 409 removes the delete action entirely — close is the only way out", async () => {
    const user = userEvent.setup();
    mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Busy Book" })] })
      .error(
        "DELETE",
        /\/api\/books\/b1\?confirm_paid=false/,
        409,
        "conflicting_job",
        "a job is live for this book",
        makeJob({ job_id: "j1", book_id: "b1", kind: "render", state: "running" }),
      );
    renderWithProviders(<Library />);

    await user.click(await screen.findByRole("button", { name: "delete" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete book" });
    await user.click(within(dialog).getByRole("button", { name: "delete book" }));

    // the refusal names the live job; deleting is impossible until it's canceled
    expect(await screen.findByText("conflicting_job")).toBeInTheDocument();
    expect(screen.getByText(/a render job is running for this book/)).toBeInTheDocument();
    // the danger key is GONE — no delete/retry/discard control remains, only close
    expect(screen.queryByRole("button", { name: "delete book" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /retry delete|discard/ })).not.toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "close" })).toBeInTheDocument();
  });

  it("a partial_delete 500 lists the surviving paths and retry re-sends confirm_paid=true", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Half Gone" })] })
      .error(
        "DELETE",
        /\/api\/books\/b1\?confirm_paid=false/,
        500,
        "partial_delete",
        "the delete only partly completed",
        { survivors: ["books/b1/normalized.json", "output/b1/render"] },
      );
    server.delete(/\/api\/books\/b1\?confirm_paid=true/, { deleted: "b1" });
    renderWithProviders(<Library />);

    await user.click(await screen.findByRole("button", { name: "delete" }));
    await user.click(await screen.findByRole("button", { name: "delete book" }));

    // every surviving path is spelled out — the user must know what may need a manual sweep
    expect(await screen.findByText("partial_delete")).toBeInTheDocument();
    expect(screen.getByText("books/b1/normalized.json")).toBeInTheDocument();
    expect(screen.getByText("output/b1/render")).toBeInTheDocument();

    // the retry must not re-trip the 402 after files are already half-gone
    await user.click(screen.getByRole("button", { name: "retry delete" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(server.lastCall("DELETE", "confirm_paid=true")).toBeDefined();
  });

  it("a live job disables delete on the card", async () => {
    mockApi().get("/api/books", {
      books: [makeBook({ book_id: "b1", active_job: { job_id: "j1", kind: "render", state: "running" } })],
    });
    renderWithProviders(<Library />);
    expect(await screen.findByRole("button", { name: "delete" })).toBeDisabled();
  });

  describe("upload result", () => {
    const ingested = (o: Record<string, unknown> = {}) => ({
      book: makeBook({ book_id: "new-book", title: "The New Novel" }),
      chapters: 12,
      blocks: 340,
      skipped_items: [],
      dropped_sections: [],
      ...o,
    });

    async function upload(container: HTMLElement) {
      const user = userEvent.setup();
      const file = new File(["fake epub bytes"], "novel.epub", { type: "application/epub+zip" });
      const input = container.querySelector<HTMLInputElement>('input[type="file"]');
      expect(input).not.toBeNull();
      await user.upload(input!, file);
      return user;
    }

    it("uploads the file as multipart form data; a 201 reads as added, with the next step", async () => {
      const server = mockApi().get("/api/books", { books: [] });
      server.post("/api/books", ingested(), 201);
      const { container } = renderWithProviders(<Library />);
      await upload(container);

      // the always-mounted live region announces just the one-line outcome
      await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/^Added The New Novel$/));
      const result = await screen.findByRole("region", { name: "Upload result" });
      expect(within(result).getByText("Added The New Novel")).toBeInTheDocument();
      expect(within(result).getByText(/12 chapters · 340 paragraphs/)).toBeInTheDocument();
      expect(within(result).queryByText(/Already in your library/)).not.toBeInTheDocument();
      expect(within(result).getByRole("link", { name: "Find the speakers" })).toHaveAttribute(
        "href",
        "/render?book=new-book#find-speakers",
      );
      expect(within(result).getByText("Nothing was left out.")).toBeInTheDocument();
      expect(within(result).queryByText(/new-book/)).not.toBeInTheDocument(); // no raw id as text
      const form = server.formBodyOf("POST", "/api/books");
      expect((form.get("file") as File).name).toBe("novel.epub");
    });

    it("a 200 (the same file again) says it is already in the library", async () => {
      mockApi().get("/api/books", { books: [] }).post("/api/books", ingested(), 200);
      const { container } = renderWithProviders(<Library />);
      await upload(container);

      await waitFor(() =>
        expect(screen.getByRole("status")).toHaveTextContent("Already in your library: The New Novel"),
      );
      const result = await screen.findByRole("region", { name: "Upload result" });
      expect(within(result).getByText("Already in your library: The New Novel")).toBeInTheDocument();
      expect(within(result).getByText(/this exact file was added before/)).toBeInTheDocument();
      // a 200 still re-reads the file on the server: never promise that nothing changed
      expect(within(result).queryByText(/nothing changed/)).not.toBeInTheDocument();
      expect(within(result).queryByText(/^Added/)).not.toBeInTheDocument();
    });

    it("the result's next step follows the live shelf: a re-upload of a failed book offers the retry", async () => {
      const shelfBook = makeBook({ book_id: "new-book", title: "The New Novel", attributed: true, assigned: true });
      mockApi()
        .get("/api/books", { books: [shelfBook] })
        .get("/api/jobs?limit=200", {
          jobs: [
            makeJob({
              book_id: "new-book",
              kind: "render",
              state: "failed",
              is_terminal: true,
              error: "GpuBusyError: another seiyuu process holds the GPU",
            }),
          ],
        })
        .post("/api/books", ingested({ book: { ...shelfBook } }), 200);
      const { container } = renderWithProviders(<Library />);
      await screen.findByRole("link", { name: "Record again" }); // the card
      await upload(container);
      const result = await screen.findByRole("region", { name: "Upload result" });
      expect(await within(result).findByRole("link", { name: "Record again" })).toHaveAttribute(
        "href",
        "/render?book=new-book#record",
      );
    });

    it("the result goes away once its book is deleted from the shelf", async () => {
      const shelfBook = makeBook({ book_id: "new-book", title: "The New Novel" });
      const server = mockApi()
        .get("/api/books", { books: [shelfBook] })
        .post("/api/books", ingested({ book: { ...shelfBook } }), 201);
      const { container, queryClient } = renderWithProviders(<Library />);
      await upload(container);
      expect(await screen.findByRole("region", { name: "Upload result" })).toBeInTheDocument();
      server.get("/api/books", { books: [] });
      await queryClient.refetchQueries({ queryKey: ["books"] });
      await waitFor(() => expect(screen.queryByRole("region", { name: "Upload result" })).not.toBeInTheDocument());
    });

    it("a failed upload is announced in the live region", async () => {
      mockApi().get("/api/books", { books: [] }).error("POST", "/api/books", 422, "unsupported", "not an epub or pdf");
      const { container } = renderWithProviders(<Library />);
      await upload(container);
      await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/Couldn't add the book: not an epub or pdf/));
      expect(screen.queryByRole("region", { name: "Upload result" })).not.toBeInTheDocument();
    });

    it("what the importer skipped sits in a closed 'What was left out' disclosure", async () => {
      mockApi()
        .get("/api/books", { books: [] })
        .post(
          "/api/books",
          ingested({
            skipped_items: ["OEBPS/copyright.xhtml", "OEBPS/toc.xhtml"],
            dropped_sections: ["Dedication (12 words)"],
          }),
          201,
        );
      const { container } = renderWithProviders(<Library />);
      const user = await upload(container);

      const summary = await screen.findByText("What was left out (3)");
      const details = summary.closest("details")!;
      expect(details).not.toHaveAttribute("open"); // evidence on demand, not in the way
      await user.click(summary);
      expect(details).toHaveAttribute("open");
      expect(within(details).getByText("OEBPS/copyright.xhtml")).toBeInTheDocument();
      expect(within(details).getByText("OEBPS/toc.xhtml")).toBeInTheDocument();
      expect(within(details).getByText("Dedication (12 words)")).toBeInTheDocument();
      expect(within(details).getByText(/front or back matter/)).toBeInTheDocument();
      expect(within(details).getByText(/too short/)).toBeInTheDocument();
      expect(screen.queryByText("Nothing was left out.")).not.toBeInTheDocument();
    });
  });

  describe("job line and failed-run banner", () => {
    const failedRender = makeJob({
      job_id: "f-123",
      book_id: "b1",
      kind: "render",
      state: "failed",
      is_terminal: true,
      error: "GpuBusyError: cannot load kokoro: another seiyuu process holds the GPU (lock: data/gpu.lock).",
      created_at: "2026-10-22T10:00:00Z",
    });

    it("the live job line is plain words, never the raw job id", async () => {
      mockApi().get("/api/books", {
        books: [makeBook({ book_id: "b1", active_job: { job_id: "job-abc123", kind: "attribute", state: "queued" } })],
      });
      renderWithProviders(<Library />);
      expect(await screen.findByText("Finding the speakers — waiting its turn")).toBeInTheDocument();
      expect(screen.queryByText(/job-abc123/)).not.toBeInTheDocument();
    });

    it("shows the latest failure with the plain fix; Dismiss hides it and is remembered by job id", async () => {
      const user = userEvent.setup();
      mockApi()
        .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Failing Book" })] })
        .get("/api/jobs?limit=200", { jobs: [failedRender] });
      const first = renderWithProviders(<Library />);

      const banner = await screen.findByRole("group", { name: "The last recording failed" });
      expect(within(banner).getByText(/Another Seiyuu program .* is using the graphics card/)).toBeInTheDocument();
      expect(within(banner).getByText("Details")).toBeInTheDocument(); // raw message disclosed, not primary
      expect(screen.queryByText(/f-123/)).not.toBeInTheDocument();

      await user.click(within(banner).getByRole("button", { name: "Dismiss" }));
      expect(screen.queryByRole("group", { name: "The last recording failed" })).not.toBeInTheDocument();
      expect(localStorage.getItem("seiyuu.dismissedFailure.f-123")).toBe("1");
      // keyboard users keep their place: focus moves to the card's first action, not <body>
      expect(document.activeElement).not.toBe(document.body);
      // (its primary next step)
      expect(document.activeElement).toHaveTextContent("Find the speakers");
      expect(document.activeElement?.closest(".bookcard .nextrow")).not.toBeNull();

      // a fresh mount (page reload) keeps it dismissed
      first.unmount();
      renderWithProviders(<Library />);
      expect(await screen.findByText("Failing Book")).toBeInTheDocument();
      expect(screen.queryByRole("group", { name: "The last recording failed" })).not.toBeInTheDocument();
    });

    it("a newer job of the same kind supersedes the failure; a different kind does not", async () => {
      const retry = makeJob({ job_id: "r2", book_id: "b1", kind: "render", state: "succeeded", is_terminal: true, created_at: "2026-10-22T11:00:00Z" });
      const server = mockApi()
        .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Retried Book" })] })
        .get("/api/jobs?limit=200", { jobs: [retry, failedRender] });
      const { queryClient } = renderWithProviders(<Library />);
      expect(await screen.findByText("Retried Book")).toBeInTheDocument();
      await waitFor(() => expect(queryClient.getQueryData(["jobs", "shelf"])).toBeDefined());
      expect(screen.queryByRole("group", { name: "The last recording failed" })).not.toBeInTheDocument();

      // only a later ATTRIBUTE run: the render failure still stands
      const attr = makeJob({ job_id: "a3", book_id: "b1", kind: "attribute", state: "succeeded", is_terminal: true, created_at: "2026-10-22T11:00:00Z" });
      server.get("/api/jobs?limit=200", { jobs: [attr, failedRender] });
      await queryClient.refetchQueries({ queryKey: ["jobs", "shelf"] });
      expect(await screen.findByRole("group", { name: "The last recording failed" })).toBeInTheDocument();
    });

    it("a retry running right now hides the banner before it reaches the shelf window", async () => {
      mockApi()
        .get("/api/books", {
          books: [
            makeBook({
              book_id: "b1",
              title: "Retrying Book",
              active_job: { job_id: "r9", kind: "render", state: "running" },
            }),
          ],
        })
        .get("/api/jobs?limit=200", { jobs: [failedRender] }); // only the old failure
      const { queryClient } = renderWithProviders(<Library />);
      expect(await screen.findByText("Retrying Book")).toBeInTheDocument();
      await waitFor(() => expect(queryClient.getQueryData(["jobs", "shelf"])).toBeDefined());
      expect(screen.queryByRole("group", { name: "The last recording failed" })).not.toBeInTheDocument();
    });

    it("dismissal survives blocked storage for the current page", async () => {
      const user = userEvent.setup();
      mockApi()
        .get("/api/books", { books: [makeBook({ book_id: "b1", title: "Private Window" })] })
        .get("/api/jobs?limit=200", { jobs: [failedRender] });
      // only the banner's keys throw: other per-viewer conveniences are not under test here
      const realGet = Storage.prototype.getItem;
      const realSet = Storage.prototype.setItem;
      const blocked = (key: string) => key.startsWith("seiyuu.dismissedFailure.");
      const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(function (this: Storage, key: string) {
        if (blocked(key)) throw new Error("SecurityError");
        return realGet.call(this, key);
      });
      const set = vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (
        this: Storage,
        key: string,
        value: string,
      ) {
        if (blocked(key)) throw new Error("SecurityError");
        realSet.call(this, key, value);
      });
      try {
        renderWithProviders(<Library />);
        const banner = await screen.findByRole("group", { name: "The last recording failed" });
        await user.click(within(banner).getByRole("button", { name: "Dismiss" }));
        expect(screen.queryByRole("group", { name: "The last recording failed" })).not.toBeInTheDocument();
      } finally {
        get.mockRestore();
        set.mockRestore();
      }
    });
  });

  describe("empty state", () => {
    it("an empty shelf shows the setup checklist above the drop zone", async () => {
      mockApi()
        .get("/api/books", { books: [] })
        .get(
          "/api/system?probe=true",
          makeSystemProbe({ ollama: { base_url: "x", reachable: true, model_installed: false } }),
        )
        .get("/api/engines", { engines: [makeEngine()] });
      const { container } = renderWithProviders(<Library />);

      const list = await screen.findByRole("list", { name: "Setup checklist" });
      expect(screen.getByText("Before your first book")).toBeInTheDocument();
      expect(await within(list).findByText("ollama pull qwen2.5:7b")).toBeInTheDocument();
      // above the drop zone, in document order
      const drop = container.querySelector(".drop")!;
      expect(list.compareDocumentPosition(drop) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });

    it("a shelf with books hides the checklist (the NavRail pill still has it)", async () => {
      mockApi().get("/api/books", { books: [makeBook({ title: "Has Books" })] });
      renderWithProviders(<Library />);
      expect(await screen.findByText("Has Books")).toBeInTheDocument();
      expect(screen.queryByRole("list", { name: "Setup checklist" })).not.toBeInTheDocument();
    });

    it("backend dies after the shelf loaded: retitled panel, Not running row, probe asked again", async () => {
      const server = mockApi()
        .get("/api/books", { books: [makeBook({ title: "Has Books" })] })
        .get("/api/system?probe=true", makeSystemProbe())
        .get("/api/engines", { engines: [makeEngine()] });
      const { queryClient } = renderWithProviders(<Library />);
      expect(await screen.findByText("Has Books")).toBeInTheDocument();
      // the probe has a cached success from before the outage
      await queryClient.fetchQuery({ queryKey: ["system", "probe"], queryFn: () => makeSystemProbe() });
      const probesBefore = server.calls.filter((c) => c.url === "/api/system?probe=true").length;

      server.on("GET", /^\/api\//, () => new Response("", { status: 502 }));
      await queryClient.refetchQueries({ queryKey: ["books"] });

      expect(await screen.findByText("The Seiyuu backend isn't running")).toBeInTheDocument();
      expect(screen.queryByText("Before your first book")).not.toBeInTheDocument();
      expect(screen.getByText(/shelf below is from before it stopped/)).toBeInTheDocument();
      const backend = await screen.findByRole("listitem", { name: "Seiyuu backend" });
      await waitFor(() => expect(within(backend).getByText("Not running")).toBeInTheDocument());
      expect(server.calls.filter((c) => c.url === "/api/system?probe=true").length).toBeGreaterThan(probesBefore);
      expect(screen.getByText("Has Books")).toBeInTheDocument(); // cached shelf stays visible
    });

    it("backend down: the checklist carries the start command instead of a raw error line", async () => {
      mockApi().on("GET", /^\/api\//, () => new Response("", { status: 500 }));
      renderWithProviders(<Library />);
      const list = await screen.findByRole("list", { name: "Setup checklist" });
      expect(await within(list).findByText("uv run uvicorn seiyuu.api.main:app")).toBeInTheDocument();
      expect(document.querySelector(".errline")).toBeNull();
    });
  });
});
