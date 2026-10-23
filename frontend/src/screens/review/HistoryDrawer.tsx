import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Dialog as AriaDialog, Heading, Modal, ModalOverlay } from "react-aria-components";

import { api, ApiError } from "../../api/client";
import { useUndoEdit } from "../../api/hooks";
import type { EditLog } from "../../api/types";
import { JobErrorText } from "../../components/JobErrorText";
import { explainJobError } from "../../lib/jobErrors";
import { describeEdit, type LineTarget } from "../../lib/reviewTools";

/** The book's edit history (GET /edits), newest first, in plain words, with a "Go to line" for
    each change that touched a line. Only the NEWEST change can be undone (the server's undo is
    DELETE /edits/last): the drawer says exactly what that removes and asks first. The change
    shown in that question is pinned, and the log is checked against it again right before
    undoing, so a change made elsewhere (another tab, a refetch) is never removed unseen. */
export function HistoryDrawer({
  bookId,
  log,
  nameOf,
  isHidden,
  onGoTo,
  onClose,
}: {
  bookId: string;
  log: EditLog | undefined;
  /** a character's name to show; null when gone; a spoiler-safe hidden one comes back hidden */
  nameOf: (id: string) => string | null;
  /** spoiler-safe hides this character: renames and merges of it leave the stored names out */
  isHidden: (id: string) => boolean;
  onGoTo: (target: LineTarget) => void;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const undo = useUndoEdit(bookId);
  /** the change "Undo this…" asked about, pinned: the confirm shows it and undo checks it */
  const [asking, setAsking] = useState<{ op: Record<string, unknown>; count: number; text: string } | null>(
    null,
  );
  const [checking, setChecking] = useState(false);
  const [changed, setChanged] = useState(false);
  const [removed, setRemoved] = useState<string | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const ops = log?.ops ?? [];
  const newest = ops.length > 0 ? ops[ops.length - 1] : null;
  const describe = (op: Record<string, unknown>) => describeEdit(op, nameOf, isHidden);

  // keyboard focus follows the controls that replace each other: "Undo it" when asked,
  // "Undo this…" (or the heading, once nothing is left) after keeping, undoing or a change
  const confirmRef = useRef<HTMLButtonElement>(null);
  const undoRef = useRef<HTMLButtonElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [focusTo, setFocusTo] = useState<"confirm" | "undo" | null>(null);
  useEffect(() => {
    if (focusTo === null) return;
    const el = focusTo === "confirm" ? confirmRef.current : (undoRef.current ?? headingRef.current);
    el?.focus();
    setFocusTo(null);
  }, [focusTo, ops.length]);

  // the history moved under an open question (a refetch, another tab): drop the question
  // rather than let it quietly point at a different change
  const pinnedStale =
    asking !== null && (ops.length !== asking.count || JSON.stringify(newest) !== JSON.stringify(asking.op));
  // while our own undo lands the log shrinks under the question on purpose: not a "change"
  const undoing = useRef(false);
  useEffect(() => {
    if (!pinnedStale || undoing.current) return;
    setAsking(null);
    setChanged(true);
    setFocusTo("undo");
  }, [pinnedStale]);

  const confirmUndo = async () => {
    const pinned = asking;
    if (!pinned) return;
    setChanged(false);
    setCheckError(null);
    setChecking(true);
    let fresh: EditLog;
    try {
      // the server removes whatever is last NOW: make sure that is still what we showed
      fresh = await qc.fetchQuery({
        queryKey: ["edits", bookId],
        queryFn: () => api<EditLog>(`/api/books/${bookId}/edits`),
        staleTime: 0,
      });
    } catch (e) {
      setCheckError(e instanceof Error ? e.message : String(e));
      return;
    } finally {
      setChecking(false);
    }
    const last = fresh.ops.length > 0 ? fresh.ops[fresh.ops.length - 1] : null;
    if (fresh.ops.length !== pinned.count || JSON.stringify(last) !== JSON.stringify(pinned.op)) {
      setAsking(null);
      setChanged(true);
      setFocusTo("undo");
      return;
    }
    undoing.current = true;
    undo.mutate(undefined, {
      onSettled: () => {
        undoing.current = false;
      },
      onSuccess: (d) => {
        setAsking(null);
        setRemoved(describe(d.removed).text);
        setFocusTo("undo");
      },
    });
  };

  const refused = undo.error
    ? explainJobError(
        undo.error instanceof ApiError ? { code: undo.error.code, message: undo.error.message } : undo.error.message,
      )
    : null;

  return (
    <ModalOverlay
      isOpen
      isDismissable
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      className="overlay on drawer-ov"
    >
      <Modal className="drawer">
        <AriaDialog className="dbody">
          <div className="dh">
            <Heading slot="title" className="dt" ref={headingRef} tabIndex={-1}>
              History
            </Heading>
            <button className="key quiet" onClick={onClose}>
              esc
            </button>
          </div>
          <div className="db">
            <p className="drawer-sub">
              Every fix you made to this book, newest first. Fixes are kept and re-applied whenever the speakers are found
              again.
            </p>
            {/* undo outcomes speak here; always mounted so screen readers hear them arrive */}
            <div aria-live="polite">
              {removed && <p className="navnote mb-2">Undone: {removed}</p>}
              {changed && (
                <p className="navnote mb-2">
                  The history changed since you opened it, so nothing was undone. Check the newest change below and try
                  again.
                </p>
              )}
            </div>
            {checkError && (
              <div className="refusal mb-2">
                <span className="tag">couldn't check the history</span>
                <p>Nothing was undone. {checkError}</p>
              </div>
            )}
            {refused && (
              <div className="refusal mb-2">
                <span className="tag">couldn't undo</span>
                <JobErrorText explained={refused} />
              </div>
            )}
            {ops.length === 0 ? (
              <p className="navnote">No fixes yet.</p>
            ) : (
              <ol className="histlist">
                {[...ops].reverse().map((op, i) => {
                  const d = describe(op);
                  const isNewest = i === 0;
                  return (
                    <li key={ops.length - i}>
                      <span className="when">
                        #{ops.length - i}
                        {isNewest ? " · newest" : ""}
                      </span>
                      <div>{d.text}</div>
                      <div className="acts">
                        {d.target && (
                          <button className="link text-[12px]" onClick={() => onGoTo(d.target as LineTarget)}>
                            Go to line
                          </button>
                        )}
                        {isNewest && !asking && (
                          <button
                            ref={undoRef}
                            className="link text-[12px]"
                            disabled={undo.isPending}
                            onClick={() => {
                              setRemoved(null);
                              setChanged(false);
                              setCheckError(null);
                              undo.reset();
                              setAsking({ op, count: ops.length, text: d.text });
                              setFocusTo("confirm");
                            }}
                          >
                            Undo this…
                          </button>
                        )}
                      </div>
                      {isNewest && asking && (
                        <div className="undoconfirm" role="group" aria-label="confirm undo">
                          <p>
                            Undo removes this change: <b>{asking.text}</b>. Everything else stays.
                          </p>
                          <div className="flex gap-2">
                            <button
                              ref={confirmRef}
                              className="key px-3 py-[3px]"
                              disabled={checking || undo.isPending}
                              onClick={() => void confirmUndo()}
                            >
                              {checking || undo.isPending ? "undoing…" : "Undo it"}
                            </button>
                            <button
                              className="key quiet px-3 py-[3px]"
                              onClick={() => {
                                setAsking(null);
                                setFocusTo("undo");
                              }}
                            >
                              Keep it
                            </button>
                          </div>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ol>
            )}
            {ops.length > 1 && (
              <p className="navnote mt-3">Only the newest change can be undone; undo it to reach the one before.</p>
            )}
          </div>
        </AriaDialog>
      </Modal>
    </ModalOverlay>
  );
}
