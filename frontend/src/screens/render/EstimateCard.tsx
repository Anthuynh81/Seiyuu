import type { CostEstimateOut, RenderMode } from "../../api/types";

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
  error: string | null;
}) {
  const totalSegs = est.paid_segments + est.cached_segments + est.free_segments || 1;
  const paid = est.total_usd > 0;
  const human = paid
    ? `Most of this render is free. ${est.paid_segments} segment(s) use a paid cloud voice — about $${est.total_usd.toFixed(2)}.`
    : est.cached_segments > 0 && est.free_segments === 0 && est.paid_segments === 0
      ? "Everything is already cached — re-rendering is instant and free."
      : "This render is entirely free: local voices only.";
  return (
    <div className="panel est mb-0">
      <div className="panel-h">
        <b>Estimate</b>
        <span className="tag ml-auto">{mode} · {est.chapters.length ? `ch ${est.chapters.join(",")}` : "whole book"}</span>
      </div>
      <div className="humanline">{human}</div>
      <div className="estbar">
        <i className="paid" style={{ width: `${(est.paid_segments / totalSegs) * 100}%` }} />
        <i className="cached" style={{ width: `${(est.cached_segments / totalSegs) * 100}%` }} />
        <i className="free" style={{ flex: 1 }} />
      </div>
      <div className="estrow"><span>paid segments</span><b>{est.paid_segments.toLocaleString()}</b></div>
      <div className="estrow"><span>cached — free to reuse</span><b>{est.cached_segments.toLocaleString()}</b></div>
      <div className="estrow"><span>free segments (local)</span><b>{est.free_segments.toLocaleString()}</b></div>
      {est.edit_warnings.length > 0 && (
        <div className="estwarn">
          <div>edit overlay — {est.edit_warnings.length} warning(s) apply to this estimate:</div>
          {est.edit_warnings.map((w) => <div key={w}>· {w}</div>)}
        </div>
      )}
      <div className="est acts">
        {paid ? (
          <button className="key" onClick={onMint} disabled={minting}>
            {minting ? "minting…" : `approve — mint quote for $${est.total_usd.toFixed(2)}`}
          </button>
        ) : (
          <button className="key" onClick={onFreeRender}>render {mode} — free, nothing to approve</button>
        )}
      </div>
      {error && <div className="errline mx-3.5 mb-3">{error}</div>}
    </div>
  );
}
