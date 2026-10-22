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
});
