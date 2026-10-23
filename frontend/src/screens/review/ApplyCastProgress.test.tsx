import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApplyCastProgress } from "./ApplyCastProgress";

const KEY = "seiyuu.castAiSeconds";

describe("ApplyCastProgress", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });
  afterEach(() => {
    vi.useRealTimers();
    localStorage.clear();
  });

  it("without AI hints shows a moving stripe and a ticking clock, no fake fraction", () => {
    render(<ApplyCastProgress useLlm={false} characters={3} />);
    const bar = screen.getByRole("progressbar", { name: "applying the cast" });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    act(() => vi.advanceTimersByTime(2000));
    expect(screen.getByText(/choosing and saving voices… 0:02/)).toBeInTheDocument();
  });

  it("with AI hints fills against the last AI apply's time and says it is an estimate", () => {
    localStorage.setItem(KEY, "40");
    render(<ApplyCastProgress useLlm characters={24} />);
    act(() => vi.advanceTimersByTime(10_000));
    const bar = screen.getByRole("progressbar", { name: /with AI hints/ });
    expect(bar).toHaveAttribute("aria-valuenow", "25");
    expect(screen.getByText(/0:10 · about 0:30 left \(estimate from the last AI apply\)/)).toBeInTheDocument();
    expect(screen.getByText(/reads your 24 characters/)).toBeInTheDocument();
  });

  it("never claims done: caps the bar and says it is still working past the estimate", () => {
    localStorage.setItem(KEY, "20");
    render(<ApplyCastProgress useLlm characters={1} />);
    act(() => vi.advanceTimersByTime(30_000));
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "95");
    expect(screen.getByText(/taking longer than last time — still working/)).toBeInTheDocument();
  });

  it("defaults to a one-minute estimate and remembers how long an AI apply took", () => {
    const { unmount } = render(<ApplyCastProgress useLlm characters={2} />);
    act(() => vi.advanceTimersByTime(15_000));
    expect(screen.getByText(/about 0:45 left/)).toBeInTheDocument();
    unmount();
    expect(localStorage.getItem(KEY)).toBe("15");
  });

  it("does not remember a fast failure or a plain apply", () => {
    const ai = render(<ApplyCastProgress useLlm characters={2} />);
    act(() => vi.advanceTimersByTime(2000));
    ai.unmount();
    const plain = render(<ApplyCastProgress useLlm={false} characters={2} />);
    act(() => vi.advanceTimersByTime(30_000));
    plain.unmount();
    expect(localStorage.getItem(KEY)).toBeNull();
  });
});
