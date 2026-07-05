/** Chapter scope for renders: whole book, or an inclusive from/to range clamped to the
    book. [] means "whole book" to every money/render endpoint. */
export type Scope = { kind: "whole" } | { kind: "range"; from: number; to: number };

export function scopeChapters(scope: Scope, total: number): number[] {
  if (scope.kind === "whole") return [];
  const from = Math.max(1, Math.min(scope.from, total));
  const to = Math.max(from, Math.min(scope.to, total));
  return Array.from({ length: to - from + 1 }, (_, i) => from + i);
}

/** The "continue where I left off" preset: the next `count` chapters starting at the
    first chapter without rendered audio. Null when the book is fully rendered or has
    no render yet (nothing to continue from). */
export function continueRange(
  renderedSet: Set<number>,
  total: number,
  count: number,
): Extract<Scope, { kind: "range" }> | null {
  if (renderedSet.size === 0 || renderedSet.size >= total) return null;
  let first = 1;
  while (first <= total && renderedSet.has(first)) first++;
  if (first > total) return null;
  return { kind: "range", from: first, to: Math.min(first + count - 1, total) };
}
