import { ApiError } from "../../api/client";
import { useSwitchRenderMode } from "../../api/hooks";
import type { ActiveJobSummary, ArchivedRenderMode, RenderSummaryOut } from "../../api/types";

/* -------------------------------------------------- active render mode (instant fallback) */

/** Which archived render manifest.json points at. Completed renders are KEPT per mode
    (manifest.single.json / manifest.multi.json); switching is an atomic pointer move on the
    server — no synthesis, no cache touch — so falling back from multivoice to single voice
    (or back) is instant and free. Only mounted once at least one mode has been rendered. */
export function ActiveModeControl({
  bookId,
  summary,
  conflict,
}: {
  bookId: string;
  summary: RenderSummaryOut;
  conflict: ActiveJobSummary | null;
}) {
  const sw = useSwitchRenderMode(bookId);
  const apiErr = sw.error instanceof ApiError ? sw.error : null;
  const modes: { mode: ArchivedRenderMode; label: string }[] = [
    { mode: "multi", label: "multivoice" },
    { mode: "single", label: "single voice" },
  ];
  return (
    <div className="mb-3.5">
      <div className="scoperow mb-0" role="group" aria-label="active render mode">
        <span className="tag">listening to</span>
        {modes.map(({ mode, label }) => {
          const onDisk = summary.available_modes.includes(mode);
          const active = summary.active_mode === mode;
          return (
            <button
              key={mode}
              className={`chap ${active ? "on" : ""}`}
              disabled={active || !onDisk || conflict !== null || sw.isPending}
              title={onDisk ? "" : `no ${label} render on disk yet — render it once to switch freely`}
              onClick={() => sw.mutate(mode)}
            >
              {label} · {active ? "active" : onDisk ? "switch" : "not rendered"}
            </button>
          );
        })}
        <span className="mono scopehint">
          {conflict
            ? `a ${conflict.kind} job is ${conflict.state} and owns the render — wait for it or cancel it first`
            : summary.available_modes.length > 1
              ? "both renders are kept — switching is instant, nothing re-renders"
              : "render the other mode once and you can switch between them instantly"}
        </span>
      </div>
      {apiErr &&
        (apiErr.code === "conflicting_job" ? (
          <div className="refusal mt-2">
            <span className="tag">conflicting_job</span>
            <p>{apiErr.message}</p>
          </div>
        ) : (
          <div className="errline mt-2">{apiErr.message}</div>
        ))}
      {sw.error && !apiErr && <div className="errline mt-2">{String(sw.error)}</div>}
    </div>
  );
}
