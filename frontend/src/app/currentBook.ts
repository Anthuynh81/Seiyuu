import { useEffect, useRef, useSyncExternalStore } from "react";
import { useSearchParams } from "react-router-dom";

import { useBooks } from "../api/hooks";
import { writeStored } from "../lib/storage";

/** The "current book": the one the workflow links (Speakers & cast, Proof, Produce, Listen)
    point at. A per-viewer convenience, never truth — an explicit ?book= in the URL always
    wins and becomes the remembered one; a remembered id is used only while that book is
    still on the shelf. Storage that throws just means it is remembered for this page only. */

const KEY = "seiyuu.currentBook";

// Used when storage refuses: keeps the convenience working for this page session. Covers
// both a read that throws (blocked site data) and a write that throws while reads still
// work (a full origin, older Safari private mode) — then the stored value is stale.
let memory: string | null = null;
let writeFailed = false;
const listeners = new Set<() => void>();

export function readRememberedBook(): string | null {
  if (writeFailed) return memory;
  // not lib/storage's readStored: a THROW (fall back to memory) must differ from "unset"
  try {
    return localStorage.getItem(KEY);
  } catch {
    return memory;
  }
}

export function rememberBook(bookId: string): void {
  if (readRememberedBook() === bookId) return;
  memory = bookId;
  if (!writeStored(KEY, bookId)) writeFailed = true;
  for (const l of listeners) l();
}

/** Test seam: forget the page-session fallback (storage itself is cleared per test). */
export function resetRememberedBookMemory(): void {
  memory = null;
  writeFailed = false;
}

/** Explicit wins; otherwise the remembered id, but only once the shelf confirms it still
    exists (an unknown shelf — still loading — confirms nothing). */
export function pickCurrentBook(
  explicit: string | null,
  remembered: string | null,
  shelfIds: readonly string[] | undefined,
): string | null {
  if (explicit) return explicit;
  if (remembered && shelfIds?.includes(remembered)) return remembered;
  return null;
}

/** `/render` + "abc" → `/render?book=abc`; no book → the bare path. */
export function withBook(path: string, bookId: string | null): string {
  return bookId ? `${path}?book=${encodeURIComponent(bookId)}` : path;
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  // another tab picking a book updates this one's links too
  const onStorage = (e: StorageEvent) => {
    if (e.key === null || e.key === KEY) onChange();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(onChange);
    window.removeEventListener("storage", onStorage);
  };
}

/** The remembered id, validated against a shelf the caller already has. */
function useRememberedBook(shelfIds: readonly string[] | undefined): string | null {
  const remembered = useSyncExternalStore(subscribe, readRememberedBook);
  return pickCurrentBook(null, remembered, shelfIds);
}

/** The book a book screen shows. An explicit ?book= wins. Without one (a typed URL, an old
    bookmark, a bare nav link), the screen settles on the remembered book — or, failing that,
    its own `fallback` — and PINS it into the URL with a replace. Pinning means:
    - the nav rail sees the book actually on screen (title, step links) and remembers it;
    - the choice is read once: a book picked in another tab can't move an open screen
      (which would drop a running chain or unsaved edits).
    `once` (Listen, whose bare URL is its shelf): pin at most once per visit, so the
    "back to shelf" button can clear ?book= without being bounced straight back. */
export function useScreenBook(
  shelfIds: readonly string[] | undefined,
  fallback: string | null | undefined,
  { once = false }: { once?: boolean } = {},
): string | null {
  const [params, setParams] = useSearchParams();
  const explicit = params.get("book");
  const remembered = useRememberedBook(shelfIds);
  // `once`: settled as soon as the URL names a book or the shelf has been checked
  const settled = useRef(!!explicit);
  const mayPin = !explicit && !(once && settled.current);
  const candidate = mayPin ? (remembered ?? fallback ?? null) : null;
  const shelfKnown = shelfIds !== undefined;

  useEffect(() => {
    if (explicit || (shelfKnown && !candidate)) settled.current = true;
    if (!candidate) return;
    settled.current = true;
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set("book", candidate);
        return next;
      },
      { replace: true },
    );
  }, [explicit, candidate, shelfKnown, setParams]);

  return explicit ?? candidate;
}

/** The nav rail's view: reads ?book= from the URL, remembers it, and resolves the current
    book. Fetches the shelf (stable ["books"] key) only when there is a remembered id to
    check, so a fresh browser makes no extra request. Returns the id and, when known, its title. */
export function useCurrentBook(): { bookId: string | null; title: string | null } {
  const [params] = useSearchParams();
  const explicit = params.get("book");
  const remembered = useSyncExternalStore(subscribe, readRememberedBook);
  const books = useBooks({ enabled: !!(explicit ?? remembered) });

  useEffect(() => {
    if (explicit) rememberBook(explicit);
  }, [explicit]);

  const shelf = books.data?.books;
  const bookId = pickCurrentBook(
    explicit,
    remembered,
    shelf?.map((b) => b.book_id),
  );
  const title = shelf?.find((b) => b.book_id === bookId)?.title ?? null;
  return { bookId, title };
}
