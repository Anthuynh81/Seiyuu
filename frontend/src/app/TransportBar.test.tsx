import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useEffect } from "react";
import { describe, expect, it } from "vitest";

import { makeBook, makeJob } from "../test/fixtures";
import { mockApi, renderWithProviders } from "../test/utils";
import { TransportBar } from "./TransportBar";
import { usePlayer } from "./usePlayer";

/** Loads one clip into the player so the /listen route mounts the audio transport. */
function LoadOneClip() {
  const player = usePlayer();
  useEffect(() => {
    player?.load("b1", "Chapter 1", [{ src: "/audio/a.wav", duration: 10, key: "a", speaker: "N", words: [] }]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

describe("TransportBar", () => {
  it("reads idle when no jobs are queued or running", async () => {
    mockApi(); // pre-registered GET /api/jobs -> {jobs: []}
    renderWithProviders(<TransportBar />);
    expect(await screen.findByText("console idle — no jobs queued or running")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "cancel" })).not.toBeInTheDocument();
  });

  it("names the book by its title and the step in plain words, with parsed chapter progress", async () => {
    mockApi()
      .get("/api/jobs", { jobs: [makeJob({ progress_text: "chapter 3/9: The Storm" })] })
      .get("/api/books", { books: [makeBook({ title: "The Hollow Crown" })] });
    renderWithProviders(<TransportBar />);
    expect(await screen.findByText("The Hollow Crown · Recording")).toBeInTheDocument();
    // never the raw "render · demo" job label
    expect(screen.queryByText(/render · demo/)).not.toBeInTheDocument();
    expect(screen.getByText("chapter 3 of 9 · 22%")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "progress" })).toHaveAttribute("aria-valuenow", "22");
    expect(screen.getByRole("button", { name: "cancel" })).toBeEnabled();
  });

  it("a chapter-range job counts within its range", async () => {
    mockApi()
      .get("/api/jobs", { jobs: [makeJob({ progress_text: "chapter 31/40: Night", params: { chapters: [30, 31, 32] } })] })
      .get("/api/books", { books: [makeBook()] });
    renderWithProviders(<TransportBar />);
    expect(await screen.findByText("chapter 31 (2 of 3) · 33%")).toBeInTheDocument();
  });

  it("an unparsable line keeps the moving stripe and shows the line as it is", async () => {
    mockApi()
      .get("/api/jobs", { jobs: [makeJob({ kind: "warmup", book_id: "engine:kokoro", progress_text: "loading kokoro weights (downloads on first use)…" })] })
      .get("/api/books", { books: [] });
    const { container } = renderWithProviders(<TransportBar />);
    expect(await screen.findByText("Kokoro · Getting the voice engine ready")).toBeInTheDocument();
    expect(screen.getByText("loading kokoro weights (downloads on first use)…")).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(container.querySelector(".meter.run")).not.toBeNull();
  });

  it("a book that is gone (or not loaded) reads as 'your book', never its id", async () => {
    mockApi().get("/api/jobs", { jobs: [makeJob({ book_id: "b-gone", kind: "master", progress_text: "" })] }).get("/api/books", { books: [] });
    renderWithProviders(<TransportBar />);
    expect(await screen.findByText("your book · Building the .m4b audiobook")).toBeInTheDocument();
    expect(screen.queryByText(/b-gone/)).not.toBeInTheDocument();
  });

  it("falls back to the queued job, which waits its turn, when nothing runs yet", async () => {
    mockApi()
      .get("/api/jobs", { jobs: [makeJob({ state: "queued", progress_text: "", started_at: null })] })
      .get("/api/books", { books: [makeBook()] });
    renderWithProviders(<TransportBar />);
    expect(await screen.findByText("waiting its turn")).toBeInTheDocument();
    expect(screen.getByText("queued")).toBeInTheDocument();
  });

  it("prefers the running job over an earlier queued one — display and cancel both target it", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/jobs", {
        jobs: [
          makeJob({ job_id: "j-q", kind: "attribute", state: "queued", progress_text: "", started_at: null }),
          makeJob({ job_id: "j-r", kind: "render", state: "running", progress_text: "chapter 3/9" }),
        ],
      })
      .get("/api/books", { books: [makeBook()] });
    server.post("/api/jobs/j-r/cancel", makeJob({ job_id: "j-r", kind: "render", cancel_requested: true }));
    renderWithProviders(<TransportBar />);

    // the bar shows the RUNNING render, not the queued attribute listed before it
    expect(await screen.findByText("Demo Book · Recording")).toBeInTheDocument();
    expect(screen.getByText("chapter 3 of 9 · 22%")).toBeInTheDocument();
    expect(screen.queryByText("Demo Book · Finding the speakers")).not.toBeInTheDocument();

    // ...and cancel targets the running job's id — never the queued one's
    await user.click(screen.getByRole("button", { name: "cancel" }));
    await waitFor(() => expect(server.lastCall("POST", "/cancel")?.url).toBe("/api/jobs/j-r/cancel"));
  });

  it("cancel POSTs /api/jobs/{id}/cancel for the shown job", async () => {
    const user = userEvent.setup();
    const server = mockApi().get("/api/jobs", { jobs: [makeJob({ job_id: "job-42" })] }).get("/api/books", { books: [] });
    server.post("/api/jobs/job-42/cancel", makeJob({ job_id: "job-42", cancel_requested: true }));
    renderWithProviders(<TransportBar />);

    await user.click(await screen.findByRole("button", { name: "cancel" }));
    await waitFor(() => expect(server.lastCall("POST", "/api/jobs/job-42/cancel")).toBeDefined());
  });

  it("a cancel-requested running job reads canceling and disables the cancel button", async () => {
    mockApi().get("/api/jobs", { jobs: [makeJob({ cancel_requested: true })] }).get("/api/books", { books: [] });
    renderWithProviders(<TransportBar />);
    expect(await screen.findByRole("button", { name: "canceling…" })).toBeDisabled();
    expect(screen.getByText("canceling")).toBeInTheDocument();
  });

  it("the speed key cycles the playback rate and persists it", async () => {
    const user = userEvent.setup();
    mockApi();
    renderWithProviders(
      <>
        <LoadOneClip />
        <TransportBar />
      </>,
      { route: "/listen" },
    );

    const key = await screen.findByRole("button", { name: "playback speed" });
    expect(key).toHaveTextContent("1×");

    await user.click(key);
    expect(key).toHaveTextContent("1.25×");
    expect(localStorage.getItem("seiyuu.rate")).toBe("1.25");

    // a full lap comes back around to 1×
    for (const expected of ["1.5×", "1.75×", "2×", "0.75×", "1×"]) {
      await user.click(key);
      expect(key).toHaveTextContent(expected);
    }
    expect(localStorage.getItem("seiyuu.rate")).toBe("1");
  });
});
