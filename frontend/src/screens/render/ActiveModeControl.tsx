import { ApiError } from "../../api/client";
import { useSwitchRenderMode } from "../../api/hooks";
import type { ActiveJobSummary, ArchivedRenderMode, RenderSummaryOut } from "../../api/types";
import { PlainError } from "../../components/PlainError";
import { stepName } from "../../lib/jobCopy";
import { ARCHIVE_LABEL } from "../../lib/render";

/* -------------------------------------------------- active render mode (instant fallback) */

/** Which recording Listen plays (and the .m4b is built from): which archived render
    manifest.json points at. Lives in the Outputs panel beside the downloads. Completed renders are KEPT per mode
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
    { mode: "multi", label: ARCHIVE_LABEL.multi },
    { mode: "single", label: ARCHIVE_LABEL.single },
  ];
  return (
    <div className="mx-3.5 my-2.5">
      <div className="scoperow mb-0" role="group" aria-label="version used for Listen">
        <span className="tag">version used for Listen</span>
        {modes.map(({ mode, label }) => {
          const onDisk = summary.available_modes.includes(mode);
          const active = summary.active_mode === mode;
          return (
            <button
              key={mode}
              className={`chap ${active ? "on" : ""}`}
              disabled={active || !onDisk || conflict !== null || sw.isPending}
              title={onDisk ? "" : `no ${label} recording yet — record it once to switch freely`}
              onClick={() => sw.mutate(mode)}
            >
              {label} · {active ? "in use" : onDisk ? "use this" : "not recorded"}
            </button>
          );
        })}
        <span className="mono scopehint">
          {conflict
            ? `${stepName(conflict.kind)} is ${conflict.state === "queued" ? "waiting its turn" : "running"} — wait for it or stop it first`
            : summary.available_modes.length > 1
              ? "both recordings are kept — switching is instant, nothing records again"
              : "record the other version once and you can switch between them instantly"}
        </span>
      </div>
      {sw.error && (
        <PlainError className="mt-2" error={sw.error} title={apiErr?.code === "conflicting_job" ? "busy" : "couldn't switch"} />
      )}
    </div>
  );
}
