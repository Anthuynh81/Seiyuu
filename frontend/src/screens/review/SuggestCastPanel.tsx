import { useState } from "react";

import { ApiError } from "../../api/client";
import type { SuggestCastResponse } from "../../api/types";

/* -------------------------------------------------- casting */

/** Preview of the smart caster: distinct voice per character. Applying is NOT a silent
    no-op — it spells out how many voices it creates and, if some are already cast, offers an
    explicit re-cast (which re-renders that audio) instead of quietly skipping them. */
export function SuggestCastPanel({
  preview,
  applying,
  applyError,
  onApply,
  onDismiss,
}: {
  preview: SuggestCastResponse;
  applying: boolean;
  /** the last apply failure — a 402 here grows the explicit paid-confirm recourse */
  applyError: ApiError | null;
  onApply: (opts: { recast: boolean; useLlm: boolean; confirmPaid?: boolean }) => void;
  onDismiss: () => void;
}) {
  const [recast, setRecast] = useState(false);
  const [useLlm, setUseLlm] = useState(false);
  const [allowPaid, setAllowPaid] = useState(false);
  const needsPaidConfirm = applyError?.code === "payment_confirmation_required";
  const create = preview.would_create_voice_ids.length;
  const existing = preview.would_recast_voice_ids.length;
  const noop = create === 0 && !recast; // every character already cast, recast off
  return (
    <div className="caststrip flex-wrap gap-2 bg-console-hi">
      <span className="tag">smart cast</span>
      <span className="mono text-[11px] text-ink-2">
        every character a distinct voice · {create} new
        {existing > 0 ? ` · ${existing} already cast` : ""}
      </span>
      {existing > 0 && (
        <label className="mono flex items-center gap-[5px] text-[11px]">
          <input type="checkbox" checked={recast} onChange={(e) => setRecast(e.target.checked)} />
          re-cast the {existing} existing (re-renders their audio)
        </label>
      )}
      <label
        className="mono flex items-center gap-[5px] text-[11px]"
        title="ask the LLM for per-character voice-trait hints. Voices stay distinct — the hint only nudges which one each character gets."
      >
        <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
        ✨ use AI trait hints
      </label>
      <span className="flex-1" />
      <button className="key quiet px-[9px] py-[3px]" onClick={onDismiss}>
        dismiss
      </button>
      <button
        className="key px-3 py-[3px]"
        disabled={applying || noop}
        title={noop ? "every character is already cast — enable re-cast to overwrite" : undefined}
        onClick={() => onApply({ recast, useLlm })}
      >
        {applying ? "applying…" : noop ? "nothing to apply" : "apply cast"}
      </button>
      {needsPaidConfirm && (
        <>
          <label className="mono flex items-center gap-[5px] text-[11px]">
            <input
              type="checkbox"
              checked={allowPaid}
              onChange={(e) => setAllowPaid(e.target.checked)}
            />
            approve the paid (Anthropic) caster
          </label>
          <button
            className="key quiet px-[9px] py-[3px]"
            disabled={!allowPaid || applying}
            onClick={() => onApply({ recast, useLlm, confirmPaid: true })}
          >
            retry cast
          </button>
        </>
      )}
    </div>
  );
}
