import { useEffect, useRef, useState } from "react";

import { readStored, writeStored } from "../../lib/storage";

/** How long the last AI-hinted apply took, so the bar's estimate fits this machine. */
const LAST_AI_SECONDS_KEY = "seiyuu.castAiSeconds";
const DEFAULT_AI_SECONDS = 60;
/** Shorter runs failed fast (AI unreachable, paid approval needed) and say nothing about speed. */
const MIN_REMEMBERED_SECONDS = 5;

function expectedAiSeconds(): number {
  const n = Number(readStored(LAST_AI_SECONDS_KEY));
  return Number.isFinite(n) && n >= MIN_REMEMBERED_SECONDS ? n : DEFAULT_AI_SECONDS;
}

function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** What an apply in flight is doing. Applying is ONE request, so there is no server-side
    progress to read: without AI hints it finishes in about a second (a moving stripe); with them
    nearly all the wait is the AI model loading onto the graphics card and then reading the cast,
    so the bar fills against how long the last AI apply took here (an estimate, said so) and the
    clock ticks so a long wait is visibly still working. */
export function ApplyCastProgress({ useLlm, characters }: { useLlm: boolean; characters: number }) {
  const started = useRef(performance.now());
  const [elapsed, setElapsed] = useState(0);
  const [expected] = useState(expectedAiSeconds);

  useEffect(() => {
    const start = started.current;
    const tick = window.setInterval(() => setElapsed((performance.now() - start) / 1000), 1000);
    return () => {
      window.clearInterval(tick);
      const took = (performance.now() - start) / 1000;
      if (useLlm && took >= MIN_REMEMBERED_SECONDS) writeStored(LAST_AI_SECONDS_KEY, String(Math.round(took)));
    };
  }, [useLlm]);

  if (!useLlm) {
    return (
      <div className="w-full" role="status">
        <div className="jobmeter run" role="progressbar" aria-label="applying the cast" />
        <div className="mono mt-1 text-[11px] text-ink-2">choosing and saving voices… {clock(elapsed)}</div>
      </div>
    );
  }

  const fraction = Math.min(0.95, elapsed / expected);
  const left = expected - elapsed;
  return (
    <div className="w-full" role="status">
      <div
        className="jobmeter"
        role="progressbar"
        aria-label="applying the cast with AI hints"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(fraction * 100)}
      >
        <i style={{ width: `${fraction * 100}%` }} />
      </div>
      <div className="mono mt-1 text-[11px] text-ink-2">
        {clock(elapsed)} ·{" "}
        {left > 0
          ? `about ${clock(left)} left (estimate from the last AI apply)`
          : "taking longer than last time — still working"}
      </div>
      <div className="mono text-[11px] text-ink-3">
        loading the AI model onto the graphics card, then it reads your {characters} character
        {characters === 1 ? "" : "s"} and suggests voice traits. Loading is most of the wait.
      </div>
    </div>
  );
}
