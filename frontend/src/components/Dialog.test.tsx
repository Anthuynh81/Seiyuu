import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { TalkDialog } from "./Dialog";

describe("TalkDialog", () => {
  it("renders an aria dialog named by its title, with body and footer", () => {
    render(
      <TalkDialog title="Delete book" onClose={() => {}} footer={<button>confirm delete</button>}>
        <p>everything under books/demo goes away</p>
      </TalkDialog>,
    );
    const dialog = screen.getByRole("dialog", { name: "Delete book" });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText("everything under books/demo goes away")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "confirm delete" })).toBeInTheDocument();
  });

  it("the esc key button and the Escape key both close a dismissable dialog", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <TalkDialog title="Rename" onClose={onClose}>
        <p>body</p>
      </TalkDialog>,
    );
    await user.click(screen.getByRole("button", { name: "esc" }));
    expect(onClose).toHaveBeenCalledTimes(1);
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("dismissable=false removes the esc affordance and ignores the Escape key", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <TalkDialog title="Rendering" onClose={onClose} dismissable={false}>
        <p>hold on</p>
      </TalkDialog>,
    );
    expect(screen.queryByRole("button", { name: "esc" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(onClose).not.toHaveBeenCalled();
  });
});
