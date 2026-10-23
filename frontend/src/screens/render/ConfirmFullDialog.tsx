import type { ReactNode } from "react";

import { TalkDialog } from "../../components/Dialog";
import type { WholeBookDetail } from "../../lib/render";

/* -------------------------------------------------- whole-book confirm (one sheet) */

/** The ONE confirmation before a long whole-book recording, asked before anything is sent.
    Free: its key starts the recording (with confirm_full). Paid: the live price ticket sits
    inside it and the ticket's confirm is the only key — it sends the cost token and
    confirm_full together, so there is never a second dialog or a refused round trip.
    A paid recording NEVER shows the free key, even while its price is being fetched again
    (an expired ticket) or after that failed. A refusal shows here, not under the overlay. */
export function ConfirmFullDialog({
  detail,
  paid,
  ticket,
  minting,
  busy,
  error,
  onConfirm,
  onRemint,
  onCancel,
}: {
  detail: WholeBookDetail;
  /** some lines use a paid voice: only a price ticket may confirm */
  paid: boolean;
  /** the QuoteTicket (live or stamped) for a paid recording; its own key confirms */
  ticket?: ReactNode;
  /** a fresh price is being fetched */
  minting: boolean;
  /** the recording request is in flight */
  busy: boolean;
  /** why the last request was refused, in plain words */
  error: string | null;
  onConfirm: () => void;
  onRemint: () => void;
  onCancel: () => void;
}) {
  const hours = detail.runtime_estimate_seconds / 3600;
  return (
    <TalkDialog
      title="Record the whole book"
      onClose={onCancel}
      footer={
        <>
          <button className="key quiet" onClick={onCancel}>not now</button>
          {!paid && (
            <button className="key" onClick={onConfirm} disabled={busy}>
              {busy ? "starting…" : "record the whole book"}
            </button>
          )}
        </>
      }
    >
      <p className="m-0 text-ink-2">
        This records all <b className="mono">{detail.speakable_blocks.toLocaleString()}</b> paragraphs
        {hours > 0 && (
          <>
            {" "}— about <b className="mono">{hours.toFixed(1)} h</b> of audio
          </>
        )}
        . It is a long job on your graphics card; you can stop it any time from the bar at the bottom, and
        running it again picks up where it stopped.
      </p>
      {paid && (
        <>
          <p className="mb-2 mt-3 text-ink-2">
            Some lines use a paid voice. This price is the most it can cost — confirming it starts the recording.
          </p>
          {ticket ??
            (minting ? (
              <div className="loadline">getting a fresh price…</div>
            ) : (
              <button className="key" onClick={onRemint}>get a fresh price</button>
            ))}
        </>
      )}
      {error && <div className="errline mb-0 mt-3">{error}</div>}
    </TalkDialog>
  );
}
