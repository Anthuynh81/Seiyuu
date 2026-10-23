import type { CostEstimateOut, RenderMode } from "../../api/types";
import { PlainErrorLine } from "../../components/PlainError";
import type { FlowFailure } from "../../lib/money";
import { MODE_LABEL } from "../../lib/render";

/* -------------------------------------------------- estimate + money flow */

export function EstimateCard({
  est,
  mode,
  onMint,
  onFreeRender,
  minting,
  error,
}: {
  est: CostEstimateOut;
  mode: RenderMode;
  onMint: () => void;
  onFreeRender: () => void;
  minting: boolean;
  error: FlowFailure | null;
}) {
  const totalSegs = est.paid_segments + est.cached_segments + est.free_segments || 1;
  const paid = est.total_usd > 0;
  const human = paid
    ? `Most of this recording is free. ${est.paid_segments} line(s) use a paid cloud voice — about $${est.total_usd.toFixed(2)}.`
    : est.cached_segments > 0 && est.free_segments === 0 && est.paid_segments === 0
      ? "Everything is already recorded — recording again is instant and free."
      : "This recording is entirely free: voices on this PC only.";
  return (
    <div className="panel est mb-0">
      <div className="panel-h">
        <b>Estimate</b>
        <span className="tag ml-auto">{MODE_LABEL[mode]} · {est.chapters.length ? `ch ${est.chapters.join(",")}` : "whole book"}</span>
      </div>
      <div className="humanline">{human}</div>
      <div className="estbar">
        <i className="paid" style={{ width: `${(est.paid_segments / totalSegs) * 100}%` }} />
        <i className="cached" style={{ width: `${(est.cached_segments / totalSegs) * 100}%` }} />
        <i className="free" style={{ flex: 1 }} />
      </div>
      <div className="estrow"><span>paid lines</span><b>{est.paid_segments.toLocaleString()}</b></div>
      <div className="estrow"><span>already recorded — free to reuse</span><b>{est.cached_segments.toLocaleString()}</b></div>
      <div className="estrow"><span>free lines (this PC)</span><b>{est.free_segments.toLocaleString()}</b></div>
      {est.edit_warnings.length > 0 && (
        <div className="estwarn">
          <div>your Review edits — {est.edit_warnings.length} warning(s) apply to this estimate:</div>
          {est.edit_warnings.map((w) => <div key={w}>· {w}</div>)}
        </div>
      )}
      <div className="est acts">
        {paid ? (
          <button className="key" onClick={onMint} disabled={minting}>
            {minting ? "getting the price…" : `get the price — about $${est.total_usd.toFixed(2)}`}
          </button>
        ) : (
          <button className="key" onClick={onFreeRender}>record — free, nothing to approve</button>
        )}
      </div>
      {error && <PlainErrorLine as="div" className="errline mx-3.5 mb-3" error={error.error} fallback={error.lead} />}
    </div>
  );
}
