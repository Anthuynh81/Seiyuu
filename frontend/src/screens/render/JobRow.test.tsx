import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { makeJob } from "../../test/fixtures";
import { renderWithProviders } from "../../test/utils";
import { JobRow } from "./JobRow";

describe("JobRow", () => {
  it("a failed row opens to the plain fix, and clicking the fix command doesn't collapse it", async () => {
    const user = userEvent.setup();
    const job = makeJob({
      kind: "attribute",
      state: "failed",
      is_terminal: true,
      error:
        "OllamaModelMissingError: model qwen2.5:7b isn't pulled in Ollama at http://localhost:11434 - run 'ollama pull qwen2.5:7b'",
    });
    const { container } = renderWithProviders(<JobRow job={job} />);
    const row = container.querySelector(".job") as HTMLElement;

    await user.click(screen.getByText("why? ▸"));
    expect(row).toHaveClass("open");
    expect(screen.getByText(/isn't downloaded yet/)).toBeInTheDocument();

    await user.click(screen.getByText("ollama pull qwen2.5:7b"));
    expect(row).toHaveClass("open");

    // the summary line still toggles it shut
    await user.click(screen.getByText("why? ▸"));
    expect(row).not.toHaveClass("open");
  });

  it("a running row reads its step in plain words and fills a bar from 'chapter N/M'", () => {
    renderWithProviders(
      <JobRow job={makeJob({ progress_text: "chapter 2/5: Arrival", params: { mode: "multivoice", chapters: [] } })} />,
    );
    expect(screen.getByText("Recording")).toBeInTheDocument();
    expect(screen.getByText("Full cast")).toBeInTheDocument();
    expect(screen.getByText("chapter 2 of 5 · 20% · Arrival")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Recording progress" })).toHaveAttribute("aria-valuenow", "20");
  });

  it("the bar holds its last chapter through other lines of the same job, and starts over for another job", () => {
    const params = { mode: "multivoice", chapters: [] };
    const job = makeJob({ job_id: "j-1", progress_text: "chapter 3/9: The Gate", params });
    const { rerender } = renderWithProviders(<JobRow job={job} />);
    const bar = () => screen.queryByRole("progressbar", { name: "Recording progress" });
    expect(bar()).toHaveAttribute("aria-valuenow", "22");

    // a notice between chapters (a flagged line) must not blink the bar back to nothing
    rerender(<JobRow job={{ ...job, progress_text: "  ! validation failed ch003_b0004[1] — retrying" }} />);
    expect(bar()).toHaveAttribute("aria-valuenow", "22");

    // a different job never inherits the old reading
    rerender(<JobRow job={makeJob({ job_id: "j-2", progress_text: "loading the voice…", params })} />);
    expect(bar()).not.toBeInTheDocument();
  });

  it("a line that isn't a chapter count shows as it is, with no bar", () => {
    renderWithProviders(<JobRow job={makeJob({ progress_text: "verifying cost approval…" })} />);
    expect(screen.getByText("verifying cost approval…")).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });
});
