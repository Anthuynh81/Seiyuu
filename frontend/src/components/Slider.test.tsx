import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { TalkSlider } from "./Slider";

/** TalkSlider is controlled; keyboard tests need the value to actually move, so wrap it
    in local state and spy on the onChange values it reports. */
function ControlledSlider({
  initial,
  min,
  max,
  step,
  spy,
}: {
  initial: number;
  min: number;
  max: number;
  step: number;
  spy?: (v: number) => void;
}) {
  const [value, setValue] = useState(initial);
  return (
    <TalkSlider
      value={value}
      onChange={(v) => {
        setValue(v);
        spy?.(v);
      }}
      min={min}
      max={max}
      step={step}
      ariaLabel="volume"
    />
  );
}

describe("TalkSlider", () => {
  it("exposes a slider with the given aria-label, current value, and default 0–100 range", () => {
    render(<TalkSlider value={40} onChange={() => {}} ariaLabel="volume" />);
    const slider = screen.getByRole("slider", { name: "volume" });
    expect(slider).toHaveValue("40");
    expect(slider).toHaveAttribute("min", "0");
    expect(slider).toHaveAttribute("max", "100");
  });

  it("ArrowRight/ArrowLeft move by `step` through onChange, clamped at max", async () => {
    const user = userEvent.setup();
    const spy = vi.fn();
    render(<ControlledSlider initial={8} min={0} max={10} step={2} spy={spy} />);
    const slider = screen.getByRole("slider", { name: "volume" });

    await user.tab();
    expect(slider).toHaveFocus();

    await user.keyboard("{ArrowRight}");
    expect(spy).toHaveBeenLastCalledWith(10);
    expect(slider).toHaveValue("10");

    await user.keyboard("{ArrowRight}"); // already at max: must stay clamped
    expect(slider).toHaveValue("10");
    expect(spy).not.toHaveBeenCalledWith(12);

    await user.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(spy).toHaveBeenLastCalledWith(6);
    expect(slider).toHaveValue("6");
  });

  it("Home jumps to min and End jumps to max", async () => {
    const user = userEvent.setup();
    render(<ControlledSlider initial={5} min={2} max={9} step={1} />);
    const slider = screen.getByRole("slider", { name: "volume" });

    await user.tab();
    await user.keyboard("{End}");
    expect(slider).toHaveValue("9");

    await user.keyboard("{Home}");
    expect(slider).toHaveValue("2");
  });
});
