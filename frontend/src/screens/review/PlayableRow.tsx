import { type ReactNode, type Ref, useEffect, useRef } from "react";

import type { VoiceOut } from "../../api/types";
import { Tip } from "../../components/Tooltip";
import { AuditionNotice } from "../voices/AuditionNotice";
import {
  type AuditionFlow,
  auditionNoticeText,
  hasAuditionNotice,
  useAuditionFlow,
} from "../voices/useAuditionFlow";

/* -------------------------------------------------- hear a voice where you cast it */

const KEY = "key quiet px-[7px] py-[2px] text-[10.5px]";

/** The compact ▶ beside a casting picker. ONE button for its whole life (▶ → … → ■), so
    keyboard focus stays on it while a sample is made and played: working is aria-disabled
    (still focusable) rather than `disabled`, which would push focus to <body>. */
function PlayKey({ flow, who, buttonRef }: { flow: AuditionFlow; who: string; buttonRef: Ref<HTMLButtonElement> }) {
  const working = flow.phase === "making" || flow.phase === "waiting";
  const label = flow.playing ? `stop ${who}` : `hear ${who}`;
  return (
    <button
      ref={buttonRef}
      className={KEY}
      aria-label={label}
      aria-busy={working || undefined}
      aria-disabled={(working && !flow.playing) || undefined}
      title={working ? "making a sample…" : label}
      onClick={() => {
        if (flow.playing) flow.stop();
        else if (!working) flow.start(false); // a fresh click is never a paid confirmation
      }}
    >
      {flow.playing ? "■" : working ? "…" : "▶"}
    </button>
  );
}

/** A casting table row whose voice cell gets a ▶ that plays `voice` speaking `text` (the
    character's own line; omitted, the server's default sentence). `who` names the voice for
    the key ("Alice's voice" → "hear Alice's voice" / "stop Alice's voice"). A refusal or a wait
    gets a full-width row underneath with the shared plain-words recourse; an always-mounted
    live region in the voice cell announces it. `cells` are the <td>s before the voice cell,
    `after` the ones following it. With no `voice` (uncast, or a voice made only on apply) the
    key is absent, or greyed with `unavailable` as its reason (a tooltip on focus and hover). */
export function PlayableRow({
  cells,
  picker,
  after,
  voice,
  text,
  who,
  unavailable,
  colSpan,
  className,
  onClick,
}: {
  cells: ReactNode;
  picker: ReactNode;
  after?: ReactNode;
  voice: VoiceOut | undefined;
  text?: string;
  who: string;
  unavailable?: string;
  colSpan: number;
  className?: string;
  onClick?: () => void;
}) {
  const flow = useAuditionFlow({
    voiceId: voice?.voice_id ?? "",
    engine: voice?.engine ?? "",
    text,
    autoplay: true,
  });
  const showNotice = voice !== undefined && hasAuditionNotice(flow);
  const announce = voice ? auditionNoticeText(flow) : null;

  // A recourse click (try again, confirm & play, Download now…) changes the notice, which can
  // remove the very button that held focus. Hand focus back to the row's ▶ rather than <body>.
  const keyRef = useRef<HTMLButtonElement>(null);
  const focusInNotice = useRef(false);
  useEffect(() => {
    if (!focusInNotice.current) return;
    const active = document.activeElement;
    if (active === null || active === document.body || !active.isConnected) {
      focusInNotice.current = false;
      keyRef.current?.focus();
    }
  });

  return (
    <>
      <tr className={className} onClick={onClick}>
        {cells}
        <td className="vcell" onClick={(e) => e.stopPropagation()}>
          <div className="flex items-center gap-1">
            <div className="min-w-0 flex-1">{picker}</div>
            {voice ? (
              <PlayKey flow={flow} who={who} buttonRef={keyRef} />
            ) : unavailable ? (
              <Tip content={unavailable}>
                <button
                  className={KEY}
                  aria-label={`hear ${who} — ${unavailable}`}
                  aria-disabled
                  onClick={(e) => e.preventDefault()}
                >
                  ▶
                </button>
              </Tip>
            ) : null}
          </div>
          {/* always mounted, so screen readers hear a refusal arrive (a region inserted
              together with its text is often not announced). A bare polite region, not
              role=status: the table's own status line stays the one "status" on the panel. */}
          <span className="sr-only" aria-live="polite" aria-atomic="true">
            {announce ?? ""}
          </span>
        </td>
        {after}
      </tr>
      {showNotice && (
        <tr className="playnote">
          <td
            colSpan={colSpan}
            onClick={(e) => e.stopPropagation()}
            onFocus={() => {
              focusInNotice.current = true;
            }}
            onBlur={(e) => {
              // focus moved somewhere on purpose; a removed button blurs with no target
              if (e.relatedTarget) focusInNotice.current = false;
            }}
          >
            <AuditionNotice flow={flow} compact live={false} />
          </td>
        </tr>
      )}
    </>
  );
}
