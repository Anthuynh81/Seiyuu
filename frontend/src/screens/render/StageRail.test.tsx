import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeBook, makeJob } from "../../test/fixtures";
import { renderWithProviders } from "../../test/utils";
import { StageRail } from "./StageRail";

// a BookCard is a superset of the rail's status payload
const status = (o: Parameters<typeof makeBook>[0] = {}) => makeBook(o);

describe("StageRail", () => {
  it("lists the steps in plain words and lights the done ones", () => {
    renderWithProviders(<StageRail status={status({ attributed: true })} activeJob={null} />);
    const steps = screen.getByRole("list", { name: "steps" });
    const items = within(steps).getAllByRole("listitem");
    expect(items.map((i) => i.textContent?.replace("›", ""))).toEqual([
      "import",
      "speakers",
      "voices",
      "record",
      "mp3s",
      "audiobook",
    ]);
    expect(items[1].querySelector(".led")).toHaveClass("ok");
    expect(items[2].querySelector(".led")).toHaveClass("off");
  });

  it("replaces the old 'next: …' hint with a real next-step button", () => {
    renderWithProviders(<StageRail status={status({ attributed: true, assigned: true })} activeJob={null} />);
    const next = screen.getByRole("link", { name: "Record the audiobook" });
    expect(next).toHaveAttribute("href", "/render?book=demo#record");
    expect(screen.queryByText(/^next:/)).not.toBeInTheDocument();
  });

  it("a running job disables the step and lights its stage", () => {
    renderWithProviders(
      <StageRail
        status={status({ attributed: true, assigned: true })}
        activeJob={{ job_id: "j-77", kind: "render", state: "running" }}
      />,
    );
    expect(screen.getByRole("button", { name: "Recording…" })).toBeDisabled();
    expect(screen.queryByText(/j-77/)).not.toBeInTheDocument();
    const record = within(screen.getByRole("list", { name: "steps" })).getByText("record");
    expect(record.querySelector(".led")).toHaveClass("run");
  });

  it("an unresolved failure turns the step into a retry with the plain fix", () => {
    renderWithProviders(
      <StageRail
        status={status()}
        activeJob={null}
        failed={makeJob({ kind: "attribute", state: "failed", error: "OllamaUnreachableError: Ollama isn't reachable" })}
      />,
    );
    const retry = screen.getByRole("link", { name: "Find the speakers again" });
    expect(retry).toHaveAttribute("href", "/render?book=demo#find-speakers");
    expect(retry).toHaveAccessibleDescription(/Start Ollama/);
  });

  it("no status yet (book loading): no button", () => {
    renderWithProviders(<StageRail status={undefined} activeJob={null} />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
