import type { JobErrorExplanation } from "../lib/jobErrors";

/** A failure explained in plain words: the sentence, the fix, a copyable command, and the raw
    message tucked under "Details" so nothing is hidden. Shared by the finish toasts, the
    Library card banner, and the Render & Jobs history. */
export function JobErrorText({ explained }: { explained: JobErrorExplanation }) {
  return (
    <div className="joberr">
      <p className="joberr-sum">{explained.summary}</p>
      {explained.fix && <p className="joberr-fix">{explained.fix}</p>}
      {explained.command && <code className="chk-cmd">{explained.command}</code>}
      {explained.details && (
        <details className="fine joberr-det">
          <summary>Details</summary>
          <div className="errblock">{explained.details}</div>
        </details>
      )}
    </div>
  );
}
