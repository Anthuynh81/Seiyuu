import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { Tip } from "./Tooltip";

/** react-aria shows tooltips on FOCUS-VISIBLE, so all tests drive focus with the
    keyboard (user.tab), never the mouse. */
function renderTip() {
  return render(
    <>
      <Tip content="Re-render this block" delay={0}>
        <button type="button">rerun</button>
      </Tip>
      <button type="button">elsewhere</button>
    </>,
  );
}

describe("Tip", () => {
  it("is absent initially and appears with its content when the child gets keyboard focus", async () => {
    const user = userEvent.setup();
    renderTip();

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    await user.tab();
    expect(screen.getByRole("button", { name: "rerun" })).toHaveFocus();
    const tip = await screen.findByRole("tooltip");
    expect(tip).toHaveTextContent("Re-render this block");
  });

  it("hides when focus leaves the wrapped child", async () => {
    const user = userEvent.setup();
    renderTip();

    await user.tab();
    await screen.findByRole("tooltip");

    await user.tab(); // focus moves to the second button → trigger blurs
    expect(screen.getByRole("button", { name: "elsewhere" })).toHaveFocus();
    await waitFor(() => expect(screen.queryByRole("tooltip")).not.toBeInTheDocument());
  });

  it("Escape dismisses the tooltip while focus stays on the trigger", async () => {
    const user = userEvent.setup();
    renderTip();

    await user.tab();
    await screen.findByRole("tooltip");

    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("tooltip")).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "rerun" })).toHaveFocus();
  });
});
