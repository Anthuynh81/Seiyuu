import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

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

  it("offers the three ways out: Review, Pronunciation (with the misheard word) and Re-make", async () => {
    const user = userEvent.setup();
    const onRemake = vi.fn();
    const row = makeRow({
      expected: "Kael'thas raised a hand.",
      transcript: "Kale has raised a hand.",
      reason: "2 of 4 words differ from the text",
    });
    renderWithProviders(<ValidationFailure bookId="demo" row={row} voiceName="Kira" onRemake={onRemake} />);
    const fixes = within(screen.getByRole("group", { name: "fix this line" }));
    expect(fixes.getByRole("link", { name: "Open in Review" })).toHaveAttribute(
      "href",
      "/review?book=demo&ch=2&block=ch002_b0007&seg=1",
    );
    expect(fixes.getByRole("link", { name: "Fix pronunciation of “Kael'thas”" })).toHaveAttribute(
      "href",
      "/lexicon?book=demo&term=Kael%27thas",
    );
    // the voice by name; the line's internal id is not the main text
    expect(screen.getByText("· Kira")).toBeInTheDocument();
    expect(screen.queryByText(/ch002_b0007/)).not.toBeInTheDocument();
    await user.click(fixes.getByRole("button", { name: "Re-make chapter 2" }));
    expect(onRemake).toHaveBeenCalledWith(2);
  });

  it("every word heard: Fix pronunciation opens the dictionary with no word; no handler, no Re-make", () => {
    renderWithProviders(
      <ValidationFailure bookId="demo" row={makeRow({ transcript: "It was twelve o'clock. Twelve." })} />,
    );
    expect(screen.getByRole("link", { name: "Fix pronunciation" })).toHaveAttribute("href", "/lexicon?book=demo");
    expect(screen.queryByRole("button", { name: /Re-make/ })).not.toBeInTheDocument();
  });

  it("a wrong number is a misread, not a word to teach: no word is suggested", () => {
    renderWithProviders(
      <ValidationFailure bookId="demo" row={makeRow({ reason: "wrong number: 'twelve' heard as 'thirteen'" })} />,
    );
    expect(screen.getByRole("link", { name: "Fix pronunciation" })).toHaveAttribute("href", "/lexicon?book=demo");
  });

  it("a word the lexicon already respells is not suggested again", () => {
    const row = makeRow({ expected: "Kale-thass raised a hand.", transcript: "Kales raised a hand." });
    renderWithProviders(<ValidationFailure bookId="demo" row={row} respelled={new Set(["kale-thass"])} />);
    expect(screen.getByRole("link", { name: "Fix pronunciation" })).toBeInTheDocument();
  });
});
