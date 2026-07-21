import { useRef, useState } from "react";

import { ApiError } from "../../api/client";
import { useDeleteCover, useUploadCover } from "../../api/hooks";
import type { CoverOut } from "../../api/types";

/* -------------------------------------------------- cover art (master + shelf) */

export function CoverControl({ bookId, cover }: { bookId: string; cover: CoverOut | null }) {
  const upload = useUploadCover(bookId);
  const remove = useDeleteCover(bookId);
  const input = useRef<HTMLInputElement>(null);
  const [confirming, setConfirming] = useState(false);
  // GET /cover is a fixed URL, so REPLACING a cover would leave the browser's cached image
  // in place — bump a ?v= buster per successful upload (the read-along's audio_key trick)
  const [bust, setBust] = useState(0);

  const submit = (file: File | undefined) => {
    if (!file) return;
    remove.reset();
    upload.mutate(file, { onSuccess: () => setBust(Date.now()) });
  };
  const err = upload.error ?? remove.error;
  const apiErr = err instanceof ApiError ? err : null;

  return (
    <div className="border-t border-hairline px-3.5 py-3">
      <div className="flex items-center gap-3">
        {cover ? (
          <img
            src={`/api/books/${bookId}/cover${bust ? `?v=${bust}` : ""}`}
            alt="cover art"
            className="h-14 w-auto border border-hairline"
          />
        ) : (
          <span aria-hidden className="h-14 w-10 shrink-0 border border-dashed border-hairline" />
        )}
        <div className="min-w-0">
          <span className="tag">cover art</span>
          <div className="mono text-[11px] text-ink-2">
            {cover
              ? `${cover.content_type} · ${Math.max(1, Math.round(cover.bytes / 1024))} KB — lands in the .m4b and on the Listen shelf`
              : "none yet — a jpeg or png here lands in the .m4b and on the Listen shelf"}
          </div>
        </div>
        <input
          ref={input}
          type="file"
          accept="image/jpeg,image/png"
          hidden
          onChange={(e) => {
            submit(e.target.files?.[0] ?? undefined);
            e.target.value = ""; // re-picking the same file (after fixing a refusal) must fire change again
          }}
        />
        <button
          className="key quiet ml-auto shrink-0"
          disabled={upload.isPending}
          onClick={() => input.current?.click()}
        >
          {upload.isPending ? "uploading…" : cover ? "replace" : "upload cover"}
        </button>
        {cover &&
          (confirming ? (
            <>
              <button
                className="key danger shrink-0"
                disabled={remove.isPending}
                onClick={() => remove.mutate(undefined, { onSettled: () => setConfirming(false) })}
              >
                {remove.isPending ? "removing…" : "really remove"}
              </button>
              <button className="key quiet shrink-0" onClick={() => setConfirming(false)}>
                keep
              </button>
            </>
          ) : (
            <button
              className="key quiet shrink-0"
              onClick={() => {
                upload.reset();
                setConfirming(true);
              }}
            >
              remove
            </button>
          ))}
      </div>
      {apiErr &&
        (apiErr.code === "conflicting_job" ? (
          <div className="refusal mt-2.5">
            <span className="tag">conflicting_job</span>
            <p>
              a master job is running and reads the cover mid-run — wait for it or cancel it in the transport bar,
              then retry.
            </p>
          </div>
        ) : (
          <div className="errline mt-2.5">{apiErr.message}</div>
        ))}
      {err && !apiErr && <div className="errline mt-2.5">{String(err)}</div>}
    </div>
  );
}
