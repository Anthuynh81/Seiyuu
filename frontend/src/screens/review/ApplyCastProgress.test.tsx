import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { rememberAiApply } from "../../lib/applyTiming";
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
    render(<ApplyCastProgress useLlm={false} />);
    const bar = screen.getByRole("progressbar", { name: "applying the cast" });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    act(() => vi.advanceTimersByTime(2000));
    expect(screen.getByText(/choosing and saving voices… 0:02/)).toBeInTheDocument();
  });

  it("with AI hints fills against the last AI apply's time and says it is an estimate", () => {
    localStorage.setItem(KEY, "40");
    render(<ApplyCastProgress useLlm />);
    act(() => vi.advanceTimersByTime(10_000));
    const bar = screen.getByRole("progressbar", { name: /with AI hints/ });
    expect(bar).toHaveAttribute("aria-valuenow", "25");
    expect(screen.getByText(/0:10 · about 0:30 left \(estimate from the last AI apply\)/)).toBeInTheDocument();
    expect(screen.getByText(/loading the AI model onto the graphics card/)).toBeInTheDocument();
  });

  it("never claims done: caps the bar and says it is still working past the estimate", () => {
    localStorage.setItem(KEY, "20");
    render(<ApplyCastProgress useLlm />);
    act(() => vi.advanceTimersByTime(30_000));
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "95");
    expect(screen.getByText(/taking longer than last time — still working/)).toBeInTheDocument();
  });

  it("defaults to a one-minute estimate and never writes the remembered time itself", () => {
    const { unmount } = render(<ApplyCastProgress useLlm />);
    act(() => vi.advanceTimersByTime(15_000));
    expect(screen.getByText(/about 0:45 left/)).toBeInTheDocument();
    unmount(); // a dismissed, canceled or failed apply must not become the next estimate
    expect(localStorage.getItem(KEY)).toBeNull();
  });

  it("rememberAiApply keeps a believable success and ignores an instant one", () => {
    rememberAiApply(2);
    expect(localStorage.getItem(KEY)).toBeNull();
    rememberAiApply(54.4);
    expect(localStorage.getItem(KEY)).toBe("54");
  });

  it("says nothing loads locally for the cloud caster", () => {
    render(<ApplyCastProgress useLlm cloud />);
    expect(screen.getByText(/the AI \(Anthropic\) is reading your characters/)).toBeInTheDocument();
    expect(screen.queryByText(/graphics card/)).not.toBeInTheDocument();
  });
});
