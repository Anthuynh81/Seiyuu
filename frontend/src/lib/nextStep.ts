import type { ActiveJobSummary, BookCard, JobKind, JobOut } from "../api/types";
import { liveJobLine } from "./jobCopy";
import { explainJobError } from "./jobErrors";

/** The ONE thing to do next for a book, in plain words, derived only from server data: the
    stage flags, the book's live job, and its last unresolved failure. Pure; the
    <NextStep/> button renders it. */
export interface NextStep {
  label: string;
  /** Router target: path + search, optionally a #anchor on the Render screen. */
  to: string;
  /** One plain sentence under the button: why this step, or what to fix first. */
  reason?: string;
  /** A job is already doing it: the button shows what is running and can't be pressed. */
  disabled?: boolean;
  /** Set when this step re-runs that failed job kind (the label reads "… again" and the reason
      is the fix), so a screen showing the failure banner can avoid saying the fix twice. */
  retryOf?: JobKind;
  /** Which step this is (unset for "add a book" and a running job), so a screen can do the
      step in place instead of following the link (the Library card opens its speaker panel). */
  stepId?: StepId;
}

export type BookFlags = Pick<
  BookCard,
  "book_id" | "ingested" | "attributed" | "assigned" | "rendered" | "assembled" | "mastered"
>;

/** Anchors on the Render screen the next-step button can land on. */
export const RENDER_ANCHOR = {
  findSpeakers: "find-speakers",
  record: "record",
  build: "build",
} as const;

export type RenderAnchor = (typeof RENDER_ANCHOR)[keyof typeof RENDER_ANCHOR];

export function bookHref(screen: "review" | "render" | "listen", bookId: string, anchor?: RenderAnchor): string {
  return `/${screen}?book=${encodeURIComponent(bookId)}${anchor ? `#${anchor}` : ""}`;
}

/** The Library's upload slot: the Library scrolls to and focuses it when this hash is set. */
export const ADD_ANCHOR = "add";

export const ADD_A_BOOK: NextStep = {
  label: "Add a book",
  to: `/#${ADD_ANCHOR}`,
  reason: "Start by adding an EPUB or PDF in the Library.",
};

export type StepId = "attribute" | "assign" | "render" | "build" | "listen";

/** Which failed job kinds a step would re-run (a failure of another kind doesn't change it). */
const RETRIES: Partial<Record<StepId, JobKind[]>> = {
  attribute: ["attribute"],
  render: ["render"],
  // the .m4b is built from the recording alone; chapter MP3s are optional, so a failed MP3
  // run never turns "Build the .m4b" into a retry
  build: ["master"],
};

function plainStep(id: StepId, bookId: string): NextStep {
  switch (id) {
    case "attribute":
      return {
        label: "Find the speakers",
        to: bookHref("render", bookId, RENDER_ANCHOR.findSpeakers),
        reason: "Seiyuu reads the book to work out who says each line. To record in one voice, skip this.",
      };
    case "assign":
      return {
        label: "Cast the voices",
        to: bookHref("review", bookId),
        reason: "Pick a voice for the narrator and each character.",
      };
    case "render":
      return {
        label: "Record the audiobook",
        to: bookHref("render", bookId, RENDER_ANCHOR.record),
        reason: "Turn the text into audio. Free voices start with one click; paid ones show the price first.",
      };
    case "build":
      return {
        label: "Build the .m4b",
        to: bookHref("render", bookId, RENDER_ANCHOR.build),
        reason: "Join the recorded chapters into one audiobook file.",
      };
    case "listen":
      return { label: "Listen", to: bookHref("listen", bookId) };
  }
}

const AGAIN: Record<"attribute" | "render" | "build", string> = {
  attribute: "Find the speakers again",
  render: "Record again",
  build: "Build the .m4b again",
};

export function nextStep({
  status,
  activeJob = null,
  failed = null,
  want,
}: {
  status: BookFlags | null | undefined;
  activeJob?: Pick<ActiveJobSummary, "kind" | "state"> | null;
  failed?: Pick<JobOut, "kind" | "error"> | null;
  /** Force this step instead of the stage-derived one (a screen about one step, e.g. Review
      for an unattributed book, must offer that step even on a book recorded in one voice).
      A running job and a book with no text still win. */
  want?: StepId;
}): NextStep {
  if (!status) return ADD_A_BOOK;
  const id = status.book_id;

  if (activeJob) {
    return {
      // the same words as the card's live job line (jobCopy), so the two can't drift apart
      label: liveJobLine({ kind: activeJob.kind, state: "running" }),
      to: bookHref("render", id),
      reason:
        activeJob.state === "queued"
          ? "It's waiting for the job ahead of it to finish."
          : "Follow it in the bar at the bottom of the screen.",
      disabled: true,
    };
  }

  if (!status.ingested) {
    return {
      label: "Add the book again",
      to: `/#${ADD_ANCHOR}`,
      reason: "This book has no readable text yet. Add its file again in the Library.",
    };
  }

  // A single-voice recording skips speakers and casting, so "recorded" wins over them.
  const step: StepId = want ?? (status.rendered
    ? status.mastered
      ? "listen"
      : "build"
    : !status.attributed
      ? "attribute"
      : !status.assigned
        ? "assign"
        : "render");
  const plain: NextStep = { ...plainStep(step, id), stepId: step };

  if (failed && RETRIES[step]?.includes(failed.kind)) {
    const explained = explainJobError(failed.error);
    return {
      ...plain,
      label: AGAIN[step as keyof typeof AGAIN],
      reason: explained.fix ?? explained.summary,
      retryOf: failed.kind,
    };
  }
  return plain;
}
