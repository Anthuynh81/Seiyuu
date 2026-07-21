import { useEffect, useState } from "react";

import type { QuoteResponse } from "../../api/types";

/* -------------------------------------------------- quote ticket */

export type TicketState =
  | { kind: "none" }
  | { kind: "live"; quote: QuoteResponse }
  | { kind: "refused"; quote: QuoteResponse; stamp: "DRIFT" | "EXPIRED" | "USED"; message: string };

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

export function QuoteTicket({
  state,
  onConfirm,
  onRemint,
  onExpired,
  busy,
}: {
  state: TicketState;
  onConfirm: () => void;
  onRemint: () => void;
  onExpired: () => void;
  busy: boolean;
}) {
  const now = useNow(state.kind === "live");
  const quote = state.kind === "none" ? null : state.quote;
  const remaining = quote ? Math.max(0, quote.expires_at - now) : 0;
  const expired = state.kind === "live" && remaining <= 0;
  useEffect(() => {
    if (expired) onExpired();
  }, [expired]); // eslint-disable-line react-hooks/exhaustive-deps
  if (state.kind === "none" || quote === null) return null;
  const dead = state.kind === "refused";
  const fusePct = (remaining / quote.ttl_seconds) * 100;
  const mmss = `${Math.floor(remaining / 60)}:${String(Math.floor(remaining % 60)).padStart(2, "0")}`;
  return (
    <div className={`paper quote ${remaining < 60 && !dead ? "hot" : ""} ${dead ? "dead" : ""}`}>
      <div className="fuse" style={{ background: `linear-gradient(90deg, currentColor 0 ${fusePct}%, transparent ${fusePct}%)` }} />
      {dead && <div className="stamp">{state.stamp}</div>}
      <div className="quote-inner">
        <div className="head">
          <span>cost quote · single use</span>
          <span style={{ color: remaining < 60 && !dead ? "#8a6d00" : undefined }}>
            {dead ? "refused" : `expires ${mmss}`}
          </span>
        </div>
        <div className="line"><span>book</span><span>{quote.book_id}</span></div>
        <div className="line"><span>chapters</span><span>{quote.chapters.length ? quote.chapters.join(", ") : "all"}</span></div>
        <div className="line"><span>paid segments</span><span>{quote.paid_segments}</span></div>
        <div className="line"><span>ceiling</span><span>${quote.max_usd_ceiling.toFixed(2)} ok</span></div>
        {dead && <div className="line text-clip"><span>refusal</span><span>{state.message}</span></div>}
        <div className="total">
          <span className="tag text-paper-ink-2">
            {dead ? "token intact — mint a fresh quote" : "the render never bills past this"}
          </span>
          <b>${quote.total_usd.toFixed(2)}</b>
        </div>
        {dead ? (
          <button className="confirm" onClick={onRemint} disabled={busy}>RE-MINT QUOTE</button>
        ) : (
          <button className="confirm" onClick={onConfirm} disabled={busy}>
            {busy ? "STARTING…" : "CONFIRM & RENDER"}
          </button>
        )}
        <div className="sig">
          <span>fingerprint {quote.fingerprint.slice(0, 4)}…{quote.fingerprint.slice(-4)}</span>
          <span>sig …{quote.token.slice(-8)}</span>
        </div>
      </div>
    </div>
  );
}
