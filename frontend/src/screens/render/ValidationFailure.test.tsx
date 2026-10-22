import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ValidationRow } from "../../api/types";
import { renderWithProviders } from "../../test/utils";
import { ValidationFailure } from "./ValidationFailure";

function makeRow(overrides: Partial<ValidationRow> = {}): ValidationRow {
  return {
    chapter_index: 2,
    block_id: "ch002_b0007",
    segment_index: 1,
    voice_id: "v-kira",
    ok: false,
    score: 0.951,
    expected: "It was twelve o'clock.",
    transcript: "It was thirteen o'clock.",
    synth_attempts: 3,
    ...overrides,
  };
}

describe("ValidationFailure", () => {
  it("says why a high-similarity read still failed, in plain words beside the score", () => {
    renderWithProviders(
      <ValidationFailure bookId="demo" row={makeRow({ reason: "wrong number: 'twelve' heard as 'thirteen'" })} />,
    );
    expect(screen.getByText("wrong number: 'twelve' heard as 'thirteen'")).toBeInTheDocument();
    expect(screen.getByText(/score 0\.95/)).toBeInTheDocument();
  });

  it("a verdict without a reason (pre-scorer, or an older backend) shows just the score", () => {
    const { container } = renderWithProviders(<ValidationFailure bookId="demo" row={makeRow()} />);
    expect(screen.getByText(/score 0\.95/)).toBeInTheDocument();
    expect(container.querySelector(".valrow > p")).toBeNull();
    // null from the API is treated the same as absent
    const { container: nulled } = renderWithProviders(
      <ValidationFailure bookId="demo" row={makeRow({ reason: null })} />,
    );
    expect(nulled.querySelector(".valrow > p")).toBeNull();
  });
});
