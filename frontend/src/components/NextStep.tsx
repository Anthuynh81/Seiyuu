import { useId } from "react";
import { Link } from "react-router-dom";

import type { NextStep as NextStepT } from "../lib/nextStep";

/** The single primary action for a book: a real link to where the next step happens, or a
    disabled key naming the job that is already doing it. The reason sits beside it as plain
    text (and describes the control for screen readers). */
export function NextStep({
  step,
  showReason = true,
  className = "",
}: {
  step: NextStepT;
  showReason?: boolean;
  className?: string;
}) {
  const reasonId = useId();
  const reason = showReason && step.reason ? step.reason : null;
  const describedBy = reason ? reasonId : undefined;
  return (
    <span className={`nextstep ${className}`}>
      {step.disabled ? (
        <button type="button" className="key" disabled aria-describedby={describedBy}>
          {step.label}
        </button>
      ) : (
        <Link className="key nextstep-key" to={step.to} aria-describedby={describedBy}>
          {step.label}
        </Link>
      )}
      {reason && (
        <span id={reasonId} className="nextstep-why">
          {reason}
        </span>
      )}
    </span>
  );
}
