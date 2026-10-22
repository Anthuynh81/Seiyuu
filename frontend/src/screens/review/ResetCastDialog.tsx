import { useSettings } from "../../api/hooks";
import type { VoiceAssignment } from "../../api/types";
import { TalkDialog } from "../../components/Dialog";
import { autoNarratorId, resetLosses } from "../../lib/casting";

/* -------------------------------------------------- casting */

/** Confirm gate for a FULL cast reset (draft with keep_existing:false). Re-draft keeps every
    hand pick by default; this is the one path that throws them away, so it spells out exactly
    what goes — each manual pick by name, a hand-chosen narrator, the thought voice — before
    anything is posted. Names resolve through the lookups, falling back to the raw ids. */
export function ResetCastDialog({
  assignment,
  characterName,
  voiceName,
  unsavedEdits,
  resetting,
  onConfirm,
  onClose,
}: {
  /** the SAVED assignment — what the server will discard */
  assignment: VoiceAssignment;
  characterName: (id: string) => string;
  voiceName: (id: string) => string;
  /** local casting edits not yet saved — the post-reset refetch overwrites them too */
  unsavedEdits: boolean;
  resetting: boolean;
  onConfirm: () => void;
  onClose: () => void;
}) {
  // The narrator is "kept" only if it is EXACTLY the one the reset writes
  // (narrator_<default preset>); until settings answer, the confirm waits, and if they fail
  // the saved narrator is listed as lost — never under-report on the data-loss gate.
  const settings = useSettings();
  const resetNarrator = settings.data ? autoNarratorId(settings.data.kokoro_default_voice) : null;
  const checking = settings.isPending;
  const { picks, narrator, thought } = resetLosses(assignment, resetNarrator);
  const nothingManual = picks.length === 0 && narrator === null && thought === null;
  return (
    <TalkDialog
      title="Reset cast"
      onClose={onClose}
      footer={
        <>
          <button className="key quiet" onClick={onClose}>
            cancel
          </button>
          <button className="key danger" disabled={resetting || checking} onClick={onConfirm}>
            {resetting ? "resetting…" : "reset cast"}
          </button>
        </>
      }
    >
      <p className="text-[12.5px]">
        Re-casts every character from scratch with auto voices and the default narrator.
      </p>
      {checking ? (
        <p className="mono mt-2 text-[11px] text-ink-2">checking what a reset would change…</p>
      ) : nothingManual ? (
        <p className="mono mt-2 text-[11px] text-ink-2">
          nothing manual to lose — every character is on its auto voice.
        </p>
      ) : (
        <div className="mt-2">
          <p className="mono text-[11px] text-ink-2">this discards:</p>
          <ul className="mono mt-1 list-disc pl-5 text-[11px]" aria-label="what a reset discards">
            {picks.length > 0 && (
              <li>
                {picks.length} manual pick{picks.length === 1 ? "" : "s"}:{" "}
                {picks
                  .map((p) => `${characterName(p.characterId)} → ${voiceName(p.voiceId)}`)
                  .join(", ")}
              </li>
            )}
            {narrator !== null && <li>narrator: {voiceName(narrator)}</li>}
            {thought !== null && <li>thought voice: {voiceName(thought)}</li>}
          </ul>
        </div>
      )}
      {unsavedEdits && (
        <p className="mono mt-2 text-[11px] text-ink-2">your unsaved casting edits are discarded too.</p>
      )}
    </TalkDialog>
  );
}
