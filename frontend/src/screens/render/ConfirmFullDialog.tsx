import { TalkDialog } from "../../components/Dialog";

/* -------------------------------------------------- confirm-full dialog */

export function ConfirmFullDialog({
  detail,
  onConfirm,
  onCancel,
}: {
  detail: { speakable_blocks: number; runtime_estimate_seconds: number };
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const hours = detail.runtime_estimate_seconds / 3600;
  return (
    <TalkDialog
      title="Full-book render"
      onClose={onCancel}
      footer={
        <>
          <button className="key quiet" onClick={onCancel}>not now</button>
          <button className="key" onClick={onConfirm}>render the whole book</button>
        </>
      }
    >
      <p className="m-0 text-ink-2">
        This renders <b className="mono">{detail.speakable_blocks.toLocaleString()}</b> segments — roughly{" "}
        <b className="mono">{hours.toFixed(1)} h</b> of audio to synthesize. A long GPU job; you can cancel it
        anytime from the transport bar and re-runs resume from the cache.
      </p>
    </TalkDialog>
  );
}
