import { explainFailure, failureLine, failureOf } from "../lib/glossary";
import { JobErrorText, RecourseButton } from "./JobErrorText";

/** Any refused request or failed load, in plain words (lib/glossary): never the raw code or
    the server's developer message as the main text — that sits under "Details". A known
    refusal brings its recourse (e.g. [Cancel recording]). `title` is an optional short lead
    ("Couldn't save"). Renders nothing for a null error. */
export function PlainError({
  error,
  title,
  className = "",
}: {
  error: unknown;
  title?: string;
  className?: string;
}) {
  if (error === null || error === undefined) return null;
  const explained = explainFailure(failureOf(error));
  return (
    <div className={`refusal ${className}`} role="alert">
      {title && <span className="tag">{title}</span>}
      <JobErrorText explained={explained} />
    </div>
  );
}

/** The one-line form for tight spots (a popover, a line note, an error line under a form):
    the plain sentence and fix together (lib/glossary failureLine), plus the recourse button
    when the refusal carries one, so [Cancel recording] survives where a full box can't fit.
    A known failure keeps the server's own words in the tooltip. `as="div"` for block spots. */
export function PlainErrorLine({
  error,
  fallback,
  className = "",
  as = "span",
}: {
  error: unknown;
  /** the lead for an unknown failure ("Couldn't save that: <server words>") */
  fallback?: string;
  className?: string;
  as?: "span" | "div";
}) {
  if (error === null || error === undefined) return null;
  const explained = explainFailure(failureOf(error));
  const Tag = as;
  return (
    <Tag className={className} role="alert" title={explained.known ? (explained.details ?? undefined) : undefined}>
      {failureLine(error, fallback)}
      {explained.recourse && (
        <>
          {" "}
          <RecourseButton recourse={explained.recourse} inline />
        </>
      )}
    </Tag>
  );
}
