import { useLocation } from "react-router-dom";

import { useCancelJob, useLiveJobs } from "../api/hooks";
import type { JobOut } from "../api/types";
import { usePlayer } from "./usePlayer";

function ledFor(job: JobOut): string {
  if (job.state === "running") return job.cancel_requested ? "cxl" : "run";
  return "q";
}

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function AudioTransport() {
  const player = usePlayer()!;
  const total = player.totalDuration;
  const before = player.clips.slice(0, player.index).reduce((a, c) => a + c.duration, 0);
  const elapsed = before + player.clipElapsed;
  const clip = player.clips[player.index];
  return (
    <div className="transport">
      <button className="playkey" onClick={() => player.toggle()} aria-label="play/pause">
        {player.playing ? "⏸" : "▶"}
      </button>
      <span className="state">
        <i className={`led ${player.playing ? "run" : "off"}`} />
        {player.playing ? "playing" : "paused"}
      </span>
      <span style={{ color: "var(--ink-2)" }}>{player.chapterTitle}</span>
      <div
        className="pmeter"
        title="click to seek"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          player.seekFraction(Math.min(Math.max((e.clientX - r.left) / r.width, 0), 0.999));
        }}
      >
        <i style={{ width: total ? `${(elapsed / total) * 100}%` : "0%" }} />
      </div>
      <span className="mono" style={{ color: "var(--ink-2)" }}>{fmt(elapsed)} / {fmt(total)}</span>
      <span className="mono" style={{ color: "var(--ink-3)" }}>{clip?.speaker ?? "—"}</span>
      <span className="vol">
        <span className="tag">vol</span>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(player.volume * 100)}
          onChange={(e) => player.setVolume(Number(e.target.value) / 100)}
          aria-label="volume"
        />
      </span>
    </div>
  );
}

/** The persistent bottom transport: the single live job on every screen, and the audio
    transport on the Listen screen (when a chapter is loaded to play). */
export function TransportBar() {
  const location = useLocation();
  const player = usePlayer();
  const cancel = useCancelJob();
  const live = useLiveJobs();

  if (location.pathname === "/listen" && player && player.clips.length > 0) {
    return <AudioTransport />;
  }

  const jobs = live.data?.jobs ?? [];
  const job = jobs.find((j) => j.state === "running") ?? jobs[0];

  if (!job) {
    return (
      <div className="transport">
        <span className="state">
          <i className="led off" />
          idle
        </span>
        <span className="idle">console idle — no jobs queued or running</span>
      </div>
    );
  }
  const canceling = job.cancel_requested && job.state === "running";
  return (
    <div className="transport">
      <span className="state">
        <i className={`led ${ledFor(job)}`} />
        {canceling ? "canceling" : job.state}
      </span>
      <span style={{ color: "var(--ink-2)" }}>
        {job.kind} · {job.book_id}
      </span>
      <div className={`meter ${job.state === "running" ? "run" : ""}`} />
      <span style={{ color: "var(--ink-2)" }}>{job.progress_text || "waiting for the worker…"}</span>
      <button
        className="cancel"
        disabled={cancel.isPending || canceling}
        onClick={() => cancel.mutate(job.job_id)}
      >
        {canceling ? "canceling…" : "cancel"}
      </button>
    </div>
  );
}
