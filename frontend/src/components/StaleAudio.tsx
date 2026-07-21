import type { BookCard } from "../api/types";

/** Post-reclone signature: assemble requires a render first, so `assembled && !rendered`
    only arises when a voice re-clone invalidated the render manifests. The chapter mp3s
    and m4b still speak with the OLD voice — they are deliberately never auto-deleted
    (possibly paid synthesis, hours of GPU work) — so say it out loud until a re-render +
    re-assemble replaces them. */
export function StaleAudioBanner({
  status,
}: {
  status: Pick<BookCard, "rendered" | "assembled"> | null | undefined;
}) {
  if (!status?.assembled || status.rendered) return null;
  return (
    <div className="caststrip flex-wrap gap-2 text-[11px]">
      <span className="tag">audio predates a voice change</span>
      <span className="mono text-ink-2">
        the assembled chapters/m4b still speak with the re-cloned voice&apos;s OLD reference —
        re-render, then re-assemble to refresh them
      </span>
    </div>
  );
}
