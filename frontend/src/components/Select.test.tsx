import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { SelectOption } from "./Select";
import { TalkSelect } from "./Select";

const ENGINES: SelectOption[] = [
  { value: "chatterbox", label: "Chatterbox" },
  { value: "kokoro", label: "Kokoro" },
  { value: "elevenlabs", label: "ElevenLabs" },
];

describe("TalkSelect", () => {
  it("shows the selected option's label on the trigger and keeps the listbox closed", () => {
    render(<TalkSelect value="kokoro" onChange={() => {}} options={ENGINES} ariaLabel="engine" />);
    expect(screen.getByRole("button")).toHaveTextContent("Kokoro");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("opens a listbox with every option; choosing one reports its value string and closes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<TalkSelect value="chatterbox" onChange={onChange} options={ENGINES} ariaLabel="engine" />);

    await user.click(screen.getByRole("button"));
    const listbox = await screen.findByRole("listbox");
    expect(within(listbox).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Chatterbox",
      "Kokoro",
      "ElevenLabs",
    ]);

    await user.click(within(listbox).getByRole("option", { name: "ElevenLabs" }));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith("elevenlabs");
    await waitFor(() => expect(screen.queryByRole("listbox")).not.toBeInTheDocument());
  });

  it("a disabled option is marked aria-disabled and cannot be chosen", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const options: SelectOption[] = [
      { value: "chatterbox", label: "Chatterbox" },
      { value: "kokoro", label: "Kokoro" },
      { value: "elevenlabs", label: "ElevenLabs", disabled: true },
    ];
    render(<TalkSelect value="chatterbox" onChange={onChange} options={options} ariaLabel="engine" />);

    await user.click(screen.getByRole("button"));
    const item = await screen.findByRole("option", { name: "ElevenLabs" });
    expect(item).toHaveAttribute("aria-disabled", "true");

    await user.click(item);
    expect(onChange).not.toHaveBeenCalled();
    // the popover must not have closed as if a choice had been made
    expect(screen.getByRole("listbox")).toBeInTheDocument();
  });

  it("is keyboard operable: Enter opens, ArrowDown navigates, Enter selects and restores focus", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<TalkSelect value="chatterbox" onChange={onChange} options={ENGINES} ariaLabel="engine" />);

    await user.tab();
    expect(screen.getByRole("button")).toHaveFocus();

    await user.keyboard("{Enter}");
    await screen.findByRole("listbox");

    await user.keyboard("{ArrowDown}{Enter}");
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith("kokoro");
    await waitFor(() => expect(screen.queryByRole("listbox")).not.toBeInTheDocument());
    expect(screen.getByRole("button")).toHaveFocus();
  });
});
