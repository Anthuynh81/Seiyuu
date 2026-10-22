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

  it("picking a file uploads it as multipart form data and reports the ingest result", async () => {
    const user = userEvent.setup();
    const server = mockApi().get("/api/books", { books: [] });
    server.post("/api/books", {
      book: makeBook({ book_id: "new-book" }),
      chapters: 12,
      blocks: 340,
    });
    const { container } = renderWithProviders(<Library />);

    const file = new File(["fake epub bytes"], "novel.epub", { type: "application/epub+zip" });
    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();
    await user.upload(input!, file);

    expect(await screen.findByText(/new-book · 12 chapters, 340 blocks/)).toBeInTheDocument();
    const form = server.formBodyOf("POST", "/api/books");
    expect((form.get("file") as File).name).toBe("novel.epub");
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
      expect(document.activeElement?.closest(".bookcard .acts")).not.toBeNull();

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
