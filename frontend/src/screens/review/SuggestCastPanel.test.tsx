import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { SuggestCastResponse, VoiceAssignment } from "../../api/types";
import { SuggestCastPanel } from "./SuggestCastPanel";

function preview(assignments: Record<string, string>): SuggestCastResponse {
  const assignment: VoiceAssignment = {
    schema_version: 1,
    book_id: "b1",
    stage: "draft",
    narrator_voice_id: "narrator_af_heart",
    assignments,
    thought_voice_id: null,
    created_at: "2026-07-11T00:00:00Z",
  };
  return {
    assignment,
    would_create_voice_ids: ["carol_auto"],
    would_recast_voice_ids: [],
    edit_warnings: [],
  };
}

function renderPanel(p: SuggestCastResponse) {
  return render(
    <SuggestCastPanel preview={p} applying={false} applyError={null} onApply={() => {}} onDismiss={() => {}} />,
  );
}

describe("SuggestCastPanel kept-picks line", () => {
  it("counts the hand picks the apply will keep (plural)", () => {
    renderPanel(preview({ alice: "af_nicole", bob: "am_adam", carol: "carol_auto" }));
    expect(screen.getByText("Your 2 picks are kept")).toBeInTheDocument();
    // the create summary still renders alongside it
    expect(screen.getByText(/1 new/)).toBeInTheDocument();
  });

  it("singular for one pick", () => {
    renderPanel(preview({ alice: "af_nicole", carol: "carol_auto" }));
    expect(screen.getByText("Your 1 pick is kept")).toBeInTheDocument();
  });

  it("no line when every character is on its auto voice", () => {
    renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }));
    expect(screen.queryByText(/kept/)).not.toBeInTheDocument();
  });
});
