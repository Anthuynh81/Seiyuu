import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  useAddBookToSeries,
  useBooks,
  useCreateSeries,
  useDraftAssignment,
  useLinkSuggestions,
  useSaveCastToSeries,
  useSeriesList,
  useUnlinkSeries,
} from "../api/hooks";
import type { LinkSuggestion, Series } from "../api/types";
import { TalkSelect } from "../components/Select";
import { applicableCount, confirmedOverrides, defaultConfirmed } from "../lib/series";

/* Series / library voice consistency (F5). Reuse a returning character's voice across the books
   of a declared series. Linking is SUGGESTION-THEN-CONFIRM: the backend only ever proposes
   within-series name matches — the user confirms which to inherit and nothing is applied
   silently (precision over recall, mirroring the alias adjudicator). Applying inherits the
   confirmed voices into the book's cast via the draft overrides seam; an explicit "save cast to
   series" writes this book's voices back so the series learns its cast progressively. */

function errText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

function bookLabel(books: { book_id: string; title: string | null }[], id: string): string {
  return books.find((b) => b.book_id === id)?.title ?? id;
}

/* -------------------------------------------------- create-from-book */

function CreateSeries({ bookId, disabled }: { bookId: string; disabled: boolean }) {
  const create = useCreateSeries();
  const [name, setName] = useState("");
  return (
    <div className="panel m-0">
      <div className="panel-h">
        <b>New series from this book</b>
      </div>
      <div className="panel-sub">
        Seeds the series with this book&apos;s cast — every assigned character&apos;s voice becomes
        a cross-book link. The book must be attributed and cast.
      </div>
      <div className="flex flex-wrap items-center gap-2.5 p-3">
        <input
          className="taginput min-w-[220px] flex-none"
          placeholder="series name (e.g. Mistborn)"
          value={name}
          aria-label="series name"
          onChange={(e) => setName(e.target.value)}
        />
        <button
          className="key"
          disabled={disabled || create.isPending || name.trim() === ""}
          onClick={() =>
            create.mutate({ name: name.trim(), bookId }, { onSuccess: () => setName("") })
          }
          title={disabled ? "cast this book first (Character Review)" : undefined}
        >
          {create.isPending ? "creating…" : "create series"}
        </button>
        {disabled && (
          <span className="mono text-[11px] text-ink-3">cast this book in Character Review first</span>
        )}
        {create.error && (
          <span className="mono text-[11px] text-clip">{errText(create.error)}</span>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------- suggestions + inherit */

function SuggestionsPanel({
  series,
  bookId,
  attributed,
}: {
  series: Series;
  bookId: string;
  attributed: boolean;
}) {
  const suggestions = useLinkSuggestions(series.series_id, bookId, attributed);
  const draft = useDraftAssignment(bookId);
  const [confirmed, setConfirmed] = useState<Set<string>>(new Set());
  const [applied, setApplied] = useState<number | null>(null);

  const rows = useMemo<LinkSuggestion[]>(() => suggestions.data?.suggestions ?? [], [suggestions.data]);
  const sig = rows.map((r) => `${r.character_id}:${r.voice_exists}`).join("|");
  useEffect(() => {
    setConfirmed(defaultConfirmed(rows));
    setApplied(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const toggle = (id: string) =>
    setConfirmed((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const inheritCount = applicableCount(rows, confirmed);

  const apply = () => {
    const overrides = confirmedOverrides(rows, confirmed);
    draft.mutate(
      { strategy: "smart", overrides },
      { onSuccess: () => setApplied(Object.keys(overrides).length) },
    );
  };

  return (
    <div className="panel m-0">
      <div className="panel-h">
        <b>Suggested cross-book links</b>
        <span className="tag ml-2">{rows.length}</span>
      </div>
      <div className="panel-sub">
        Name matches within <b>{series.name}</b> — confirm which returning characters inherit their
        series voice. Nothing is applied until you click apply; a deleted voice can&apos;t be
        inherited.
      </div>

      {!attributed && (
        <div className="refusal m-3">
          <span className="tag">stage_prerequisite</span>
          <p>this book has no attribution yet — run it from Render &amp; Jobs to match characters</p>
        </div>
      )}
      {suggestions.isPending && attributed && (
        <div className="loadline p-3.5">matching returning characters…</div>
      )}
      {suggestions.error && (
        <div className="refusal m-3">
          <span className="tag">error</span>
          <p>{errText(suggestions.error)}</p>
        </div>
      )}
      {suggestions.data && rows.length === 0 && (
        <div className="sub p-3.5">
          No returning characters matched by name in this series yet. Cast another book and use
          “save cast to series” to teach it, then a sibling book will match here.
        </div>
      )}

      {rows.length > 0 && (
        <>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="mono text-left text-[11px] text-ink-3">
                  <th className="px-2.5 py-1 font-semibold">inherit</th>
                  <th className="px-1.5 py-1 font-semibold">character</th>
                  <th className="px-1.5 py-1 font-semibold">series voice</th>
                  <th className="px-1.5 py-1 font-semibold">status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.character_id} className={r.voice_exists ? "" : "opacity-55"}>
                    <td className="px-2.5 py-1 text-center">
                      <input
                        type="checkbox"
                        checked={confirmed.has(r.character_id) && r.voice_exists}
                        disabled={!r.voice_exists}
                        aria-label={`inherit ${r.canonical_name}`}
                        onChange={() => toggle(r.character_id)}
                      />
                    </td>
                    <td className="px-1.5 py-1">{r.canonical_name}</td>
                    <td className="mono px-1.5 py-1 text-[11px] text-ink-2">{r.voice_id}</td>
                    <td className="mono px-1.5 py-1 text-[11px]">
                      {r.voice_exists ? (
                        <span className="text-ink-3">available</span>
                      ) : (
                        <span className="text-clip">voice deleted</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap items-center gap-3 p-3">
            <button
              className="key"
              disabled={draft.isPending || inheritCount === 0}
              onClick={apply}
              title="inherit the confirmed voices into this book's cast and re-cast the rest collision-free"
            >
              {draft.isPending ? "applying…" : `apply ${inheritCount} inherited voice(s)`}
            </button>
            <span className="mono text-[11px] text-ink-2">
              inherits the confirmed voices, then casts the remaining characters distinct
            </span>
            {applied !== null && !draft.isPending && (
              <span className="mono text-[11px] text-ok">
                applied · {applied} inherited into this book&apos;s cast (review in Character Review)
              </span>
            )}
            {draft.error && <span className="mono text-[11px] text-clip">{errText(draft.error)}</span>}
          </div>
        </>
      )}
    </div>
  );
}

/* -------------------------------------------------- series detail */

function SeriesDetail({
  series,
  bookId,
  books,
  attributed,
  assigned,
}: {
  series: Series;
  bookId: string;
  books: { book_id: string; title: string | null }[];
  attributed: boolean;
  assigned: boolean;
}) {
  const addBook = useAddBookToSeries(series.series_id);
  const saveCast = useSaveCastToSeries(series.series_id);
  const unlink = useUnlinkSeries(series.series_id);
  const [saved, setSaved] = useState<number | null>(null);

  const isMember = series.book_ids.includes(bookId);
  const links = Object.entries(series.voice_links);

  return (
    <>
      <div className="panel m-0">
        <div className="panel-h">
          <b>{series.name}</b>
          <span className="tag ml-2">{series.book_ids.length} book(s)</span>
          <span className="tag">{links.length} linked voice(s)</span>
        </div>

        <div className="panel-sub">
          <b>Books in this series</b>
        </div>
        <div className="flex flex-wrap gap-2 px-3 py-2.5">
          {series.book_ids.length === 0 && <span className="sub m-0">no books yet</span>}
          {series.book_ids.map((id) => (
            <span
              key={id}
              className={`chip cursor-default ${id === bookId ? "border-tungsten" : ""}`}
            >
              {bookLabel(books, id)}
              {id === bookId && <span className="mono text-ink-3"> · current</span>}
            </span>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-3 px-3 pb-3">
          {!isMember ? (
            <>
              <button
                className="key"
                disabled={addBook.isPending}
                onClick={() => addBook.mutate(bookId)}
                title="attach this book so its returning characters can inherit series voices"
              >
                {addBook.isPending ? "adding…" : "add this book to the series"}
              </button>
              <span className="mono text-[11px] text-ink-3">
                adding only joins membership — it never learns this book&apos;s cast silently
              </span>
            </>
          ) : (
            <>
              <button
                className="key quiet"
                disabled={saveCast.isPending || !assigned}
                onClick={() =>
                  saveCast.mutate(bookId, { onSuccess: (r) => setSaved(r.linked_keys.length) })
                }
                title={
                  assigned
                    ? "fold this book's character voices back into the series links"
                    : "cast this book first (Character Review)"
                }
              >
                {saveCast.isPending ? "saving…" : "save cast to series"}
              </button>
              <span className="mono text-[11px] text-ink-2">
                explicit write-back — this book&apos;s voices become series links (last-write-wins)
              </span>
              {saved !== null && !saveCast.isPending && (
                <span className="mono text-[11px] text-ok">saved · {saved} link(s) added / updated</span>
              )}
            </>
          )}
          {(addBook.error || saveCast.error) && (
            <span className="mono text-[11px] text-clip">{errText(addBook.error ?? saveCast.error)}</span>
          )}
        </div>
      </div>

      <div className="h-4" />
      <SuggestionsPanel series={series} bookId={bookId} attributed={attributed} />

      <div className="h-4" />
      <div className="panel m-0">
        <div className="panel-h">
          <b>Voice links</b>
          <span className="tag ml-2">{links.length}</span>
        </div>
        <div className="panel-sub">
          The series&apos; learned cast — one voice per returning character, keyed by name. Unlink
          to stop a character inheriting across the series.
        </div>
        {links.length === 0 ? (
          <div className="sub p-3.5">
            no voice links yet — “save cast to series” on a cast book teaches the series its voices
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <tbody>
                {links.map(([key, voiceId]) => (
                  <tr key={key}>
                    <td className="px-2.5 py-1">{key}</td>
                    <td className="mono px-1.5 py-1 text-[11px] text-ink-2">{voiceId}</td>
                    <td className="px-2.5 py-1 text-right">
                      <button
                        className="key quiet px-2 py-[2px]"
                        disabled={unlink.isPending}
                        onClick={() => unlink.mutate(key)}
                        aria-label={`unlink ${key}`}
                        title="remove this cross-book link"
                      >
                        unlink
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {unlink.error && (
          <div className="mono px-3 pb-3 text-[11px] text-clip">{errText(unlink.error)}</div>
        )}
      </div>
    </>
  );
}

/* -------------------------------------------------- the screen */

export function Series() {
  const [params, setParams] = useSearchParams();
  const books = useBooks();
  const seriesList = useSeriesList();

  const attributedBooks = books.data?.books.filter((b) => b.attributed) ?? [];
  const pickBooks = attributedBooks.length > 0 ? attributedBooks : (books.data?.books ?? []);
  const bookId = params.get("book") ?? pickBooks[0]?.book_id ?? null;
  const currentBook = pickBooks.find((b) => b.book_id === bookId);
  const attributed = !!currentBook?.attributed;
  const assigned = !!currentBook?.assigned;

  const allSeries = useMemo(() => seriesList.data?.series ?? [], [seriesList.data]);
  const selectedId = params.get("series");
  const selected = allSeries.find((s) => s.series_id === selectedId) ?? null;

  const setBook = (id: string) => setParams((p) => { p.set("book", id); return p; });
  const setSeries = (id: string | null) => {
    setParams((p) => {
      if (id) p.set("series", id);
      else p.delete("series");
      return p;
    });
  };

  if (books.isPending) {
    return (
      <section className="screen">
        <div className="loadline">reading the shelf…</div>
      </section>
    );
  }
  if (!bookId) {
    return (
      <section className="screen">
        <h1>Series</h1>
        <p className="sub">No books yet — add one from the Library first.</p>
      </section>
    );
  }

  return (
    <section className="screen">
      <h1 className="flex items-baseline gap-3.5">
        Series
        <TalkSelect
          className="bookpick"
          ariaLabel="book"
          value={bookId}
          onChange={setBook}
          options={pickBooks.map((b) => ({ value: b.book_id, label: b.title ?? b.book_id }))}
        />
      </h1>
      <p className="sub">
        Keep a character&apos;s voice consistent across the books of a series. Cross-book linking is
        suggestion-then-confirm: matches are scoped to a declared series and surfaced for you to
        confirm — two same-named characters in unrelated books are never merged onto one voice.
      </p>

      <div className="drainstrip">
        <span className="tag">series</span>
        <TalkSelect
          className="bookpick"
          ariaLabel="series"
          value={selectedId ?? "__none__"}
          onChange={(v) => setSeries(v === "__none__" ? null : v)}
          options={[
            { value: "__none__", label: "— pick a series —" },
            ...allSeries.map((s) => ({ value: s.series_id, label: `${s.name} (${s.book_ids.length})` })),
          ]}
        />
        <span className="mono text-[11px] text-ink-2">{allSeries.length} series in the library</span>
        {seriesList.error && (
          <span className="mono text-[11px] text-clip">{errText(seriesList.error)}</span>
        )}
      </div>

      {!selected ? (
        <CreateSeries bookId={bookId} disabled={!assigned} />
      ) : (
        <>
          <SeriesDetail
            series={selected}
            bookId={bookId}
            books={pickBooks}
            attributed={attributed}
            assigned={assigned}
          />
          <div className="h-4" />
          <CreateSeries bookId={bookId} disabled={!assigned} />
        </>
      )}
    </section>
  );
}
