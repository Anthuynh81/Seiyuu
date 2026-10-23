import { useEffect, useState } from "react";

import { expectedAiSeconds } from "../../lib/applyTiming";

function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** What an apply in flight is doing. Applying is ONE request, so there is no server-side
    progress to read: without AI hints it finishes in about a second (a moving stripe); with them
    nearly all the wait is the AI model loading onto the graphics card and then reading the cast,
    so the bar fills against how long the last successful AI apply took here (an estimate, said
    so) and the clock ticks so a long wait is visibly still working. No live region: a clock
    announced every second would drown a screen reader; the progressbar carries the state.
    ``cloud`` is the paid Anthropic caster: nothing loads on this machine then. */
export function ApplyCastProgress({ useLlm, cloud = false }: { useLlm: boolean; cloud?: boolean }) {
  const [started] = useState(() => performance.now());
  const [elapsed, setElapsed] = useState(0);
  const [expected] = useState(expectedAiSeconds);

  useEffect(() => {
    const tick = window.setInterval(() => setElapsed((performance.now() - started) / 1000), 1000);
    return () => window.clearInterval(tick);
  }, [started]);

  if (!useLlm) {
    return (
      <div className="w-full">
        <div className="jobmeter run" role="progressbar" aria-label="applying the cast" />
        <div className="mono mt-1 text-[11px] text-ink-2">choosing and saving voices… {clock(elapsed)}</div>
      </div>
    );
  }

  const fraction = Math.min(0.95, elapsed / expected);
  const left = expected - elapsed;
  return (
    <div className="w-full">
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
        {cloud
          ? "the AI (Anthropic) is reading your characters and suggesting voice traits."
          : "loading the AI model onto the graphics card, then it reads your characters and suggests voice traits. Loading is most of the wait."}
      </div>
    </div>
  );
}
