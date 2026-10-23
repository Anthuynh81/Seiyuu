import { useCancelJob } from "../api/hooks";
import type { JobErrorExplanation, Recourse } from "../lib/glossary";

/** The one action a failure offers, when the UI can do it (e.g. [Cancel recording] for the
    job in the way). A click acts once; it never retries the refused change on its own. */
export function RecourseButton({ recourse, inline = false }: { recourse: Recourse; inline?: boolean }) {
  const cancel = useCancelJob();
  // inline: inside a one-line note (a popover's <span>, a line note) where a <p> can't go
  const Wrap = inline ? "span" : "p";
  if (cancel.isSuccess) {
    return <Wrap className="joberr-fix">Stopping it — try again once it has stopped.</Wrap>;
  }
  return (
    <Wrap className="joberr-fix">
      <button className="link" disabled={cancel.isPending} onClick={() => cancel.mutate(recourse.jobId)}>
        {cancel.isPending ? "stopping…" : recourse.label}
      </button>
      {cancel.isError && <span className="ml-2 text-clip">That didn't work — try the bar at the bottom.</span>}
    </Wrap>
  );
}

/** A failure explained in plain words: the sentence, the fix, a copyable command, a recourse
    button where there is one, and the raw message tucked under "Details" so nothing is hidden.
    Shared by the finish toasts, the Library card banner, the Produce history and every screen
    that shows a refused request (via PlainError). */
export function JobErrorText({ explained }: { explained: JobErrorExplanation }) {
  return (
    <div className="joberr">
      <p className="joberr-sum">{explained.summary}</p>
      {explained.fix && <p className="joberr-fix">{explained.fix}</p>}
      {explained.command && <code className="chk-cmd">{explained.command}</code>}
      {explained.recourse && <RecourseButton recourse={explained.recourse} />}
      {explained.details && (
        <details className="fine joberr-det">
          <summary>Details</summary>
          <div className="errblock">{explained.details}</div>
        </details>
      )}
    </div>
  );
}
