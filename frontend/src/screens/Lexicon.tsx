import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  useBooks,
  useLexicon,
  usePreviewLexicon,
  useSaveLexicon,
  useSuggestRespellings,
} from "../api/hooks";
import type { LexiconEntry, SuggestedTerm } from "../api/types";
import { TalkSelect } from "../components/Select";
import { applyRespellings, blankEntry, cleanForSave, duplicateTerms, entriesSig } from "../lib/lexicon";

/* A per-book pronunciation dictionary: term -> respelling (spoken on every engine), with an
   optional IPA that only the Kokoro engine uses. Editing re-synthesizes only the segments whose
   text changes; the affected-segment count is shown before and after a save. */

function EntryRow({
  entry,
  onChange,
  onRemove,
  duplicate,
}: {
  entry: LexiconEntry;
  onChange: (next: LexiconEntry) => void;
  onRemove: () => void;
  duplicate: boolean;
}) {
  const set = (patch: Partial<LexiconEntry>) => onChange({ ...entry, ...patch });
  return (
    <tr>
      <td>
        <input
          className={`taginput ${duplicate ? "border-clip" : ""}`}
          value={entry.term}
          placeholder="as written"
          aria-label="term"
          onChange={(e) => set({ term: e.target.value })}
        />
      </td>
      <td>
        <input
          className="taginput"
          value={entry.respelling}
          placeholder="say it like this"
          aria-label="respelling"
          onChange={(e) => set({ respelling: e.target.value })}
        />
      </td>
      <td>
        <input
          className="taginput"
          value={entry.ipa ?? ""}
          placeholder="IPA (Kokoro only)"
          aria-label="ipa"
          onChange={(e) => set({ ipa: e.target.value || null })}
        />
      </td>
      <td>
        <input
          className="taginput"
          value={entry.note ?? ""}
          placeholder="note"
          aria-label="note"
          onChange={(e) => set({ note: e.target.value || null })}
        />
      </td>
      <td className="text-center">
        <input
          type="checkbox"
          checked={entry.case_sensitive}
          aria-label="case sensitive"
          title="match the exact capitalization"
          onChange={(e) => set({ case_sensitive: e.target.checked })}
        />
      </td>
      <td>
        <button className="key quiet" onClick={onRemove} aria-label="remove">
          ✕
        </button>
      </td>
    </tr>
  );
}

export function Lexicon() {
  const [params, setParams] = useSearchParams();
  const books = useBooks();
  const ingested = books.data?.books.filter((b) => b.ingested) ?? [];
  const bookId = params.get("book") ?? ingested[0]?.book_id ?? null;

  const lexicon = useLexicon(bookId);
  const save = useSaveLexicon(bookId ?? "");
  const preview = usePreviewLexicon(bookId ?? "");
  const suggestAI = useSuggestRespellings(bookId ?? "");

  const [rows, setRows] = useState<LexiconEntry[]>([]);
  const [savedInfo, setSavedInfo] = useState<{ affected: number; total: number } | null>(null);
  const [allowPaid, setAllowPaid] = useState(false);

  // Re-seed the editable rows whenever the server copy (or the selected book) changes.
  const serverSig = lexicon.data ? entriesSig(lexicon.data.entries) : null;
  const seededBook = useRef<string | null>(null);
  useEffect(() => {
    const sameBook = seededBook.current === bookId;
    seededBook.current = bookId;
    // A save invalidates the lexicon query, and the refetch hands back the editor's own
    // rows — that is the save LANDING, not an external change. Skip the reset so the
    // "saved · N of M" confirmation survives its own round-trip (it used to be cleared
    // the same frame it became visible). A book switch always reseeds.
    if (sameBook && lexicon.data && serverSig === entriesSig(rows)) return;
    if (lexicon.data) setRows(lexicon.data.entries.map((e) => ({ ...e })));
    setSavedInfo(null);
    preview.reset();
    suggestAI.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bookId, serverSig]);

  const dupes = duplicateTerms(rows);
  const dirty = serverSig !== null && entriesSig(rows) !== serverSig;
  const canSave = dirty && dupes.length === 0 && !save.isPending;

  const suggestions = useMemo<SuggestedTerm[]>(() => {
    const have = new Set(rows.map((r) => r.term.trim().toLowerCase()).filter(Boolean));
    return (lexicon.data?.suggestions ?? []).filter((s) => !have.has(s.term.toLowerCase()));
  }, [lexicon.data, rows]);

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
        <h1>Pronunciation</h1>
        <p className="sub">No ingested books yet — add one from the Library first.</p>
      </section>
    );
  }

  const update = (i: number, next: LexiconEntry) =>
    setRows((rs) => rs.map((r, j) => (j === i ? next : r)));
  const remove = (i: number) => setRows((rs) => rs.filter((_, j) => j !== i));
  const addRow = (term = "") => setRows((rs) => [...rs, blankEntry(term)]);

  const onSave = () => {
    save.mutate(cleanForSave(rows), {
      onSuccess: (res) => {
        setSavedInfo({ affected: res.affected_blocks, total: res.total_speakable_blocks });
        preview.reset(); // the previewed delta just landed — don't show both counts
      },
    });
  };

  // F3: ask the LLM to propose respellings. Send the terms currently in the editor; if there are
  // none, an empty list lets the backend fall back to its deterministic hard-name suggestions.
  // The result is ADVISORY — folded into the rows (never clobbering a respelling the user typed),
  // still requiring an explicit Save.
  const onSuggestAI = () => {
    const terms = cleanForSave(
      rows.filter((r) => r.term.trim()).map((r) => ({ ...r, respelling: r.respelling || "x" })),
    ).map((r) => r.term);
    suggestAI.mutate(
      { terms, confirm_paid: allowPaid },
      { onSuccess: (res) => setRows((rs) => applyRespellings(rs, res.suggestions)) },
    );
  };

  const aiError = suggestAI.error instanceof ApiError ? suggestAI.error : null;
  const needsPaidConfirm = aiError?.status === 402;

  return (
    <section className="screen">
      <h1 className="flex items-baseline gap-3.5">
        Pronunciation
        <TalkSelect
          className="bookpick"
          ariaLabel="book"
          value={bookId}
          onChange={(v) => setParams({ book: v })}
          options={ingested.map((b) => ({ value: b.book_id, label: b.title ?? b.book_id }))}
        />
      </h1>
      <p className="sub">
        Fix mispronounced names and invented words for this book. A respelling is spoken by every
        engine; an optional IPA is used only by Kokoro. Editing re-synthesizes only the segments
        whose text changes.
      </p>

      {lexicon.isError && (
        <div className="refusal">
          <span className="tag">error</span>
          <p>{(lexicon.error as Error).message}</p>
        </div>
      )}

      <div className="panel m-0">
        <div className="panel-h">
          <b>Entries</b>
          <span className="tag ml-2">{cleanForSave(rows).length}</span>
        </div>
        <div className="overflow-x-auto px-3.5 pt-2">
          <table className="w-full border-collapse">
            <thead>
              <tr className="mono text-left text-[11px] text-ink-3">
                <th className="px-1.5 py-1 font-semibold">term</th>
                <th className="px-1.5 py-1 font-semibold">respelling</th>
                <th className="px-1.5 py-1 font-semibold">ipa (kokoro)</th>
                <th className="px-1.5 py-1 font-semibold">note</th>
                <th className="px-1.5 py-1 font-semibold">case</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((e, i) => (
                <EntryRow
                  key={i}
                  entry={e}
                  duplicate={dupes.includes(e.term)}
                  onChange={(next) => update(i, next)}
                  onRemove={() => remove(i)}
                />
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={6} className="sub px-1.5 py-2.5">
                    no entries yet — add a term below or pick from suggestions
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2.5 px-3.5 pb-3.5">
          <button className="key quiet" onClick={() => addRow()}>
            + add term
          </button>
          <button
            className="key quiet"
            disabled={!dirty || dupes.length > 0 || preview.isPending}
            onClick={() => preview.mutate(cleanForSave(rows))}
            title="how many segments this change would re-synthesize"
          >
            preview impact
          </button>
          <button className="key" disabled={!canSave} onClick={onSave}>
            {save.isPending ? "saving…" : "save"}
          </button>

          <span className="flex-1" />
          <button
            className="key quiet"
            disabled={suggestAI.isPending}
            onClick={onSuggestAI}
            title="ask the LLM to propose grapheme respellings — for the terms above, or (if none) for the book's likely hard names. Advisory: you still review and save."
          >
            {suggestAI.isPending ? "asking AI…" : "✨ suggest with AI"}
          </button>

          {dupes.length > 0 && (
            <span className="mono text-[11px] text-clip">duplicate term: {dupes.join(", ")}</span>
          )}
          {preview.data && !preview.isPending && (
            <span className="mono text-[11px] text-ink-2">
              would re-synthesize {preview.data.affected_blocks} of {preview.data.total_speakable_blocks} segments
            </span>
          )}
          {savedInfo && !dirty && (
            <span className="mono text-[11px] text-ink-2">
              saved · {savedInfo.affected} of {savedInfo.total} segments will re-synthesize on next render
            </span>
          )}
          {save.isError && (
            <span className="mono text-[11px] text-clip">{(save.error as Error).message}</span>
          )}
          {suggestAI.data && !suggestAI.isPending && (
            <span className="mono text-[11px] text-ink-2">
              {suggestAI.data.suggestions.length} respelling
              {suggestAI.data.suggestions.length === 1 ? "" : "s"} from {suggestAI.data.provider}/
              {suggestAI.data.model} — review and save
            </span>
          )}
          {aiError && <span className="mono text-[11px] text-clip">{aiError.message}</span>}
        </div>

        {needsPaidConfirm && (
          <div className="mt-2 flex flex-wrap items-center gap-2 px-3.5 pb-3.5">
            <label className="mono flex items-center gap-[5px] text-[11px]">
              <input
                type="checkbox"
                checked={allowPaid}
                onChange={(e) => setAllowPaid(e.target.checked)}
              />
              approve the paid (Anthropic) suggester
            </label>
            <button
              className="key quiet px-[9px] py-[3px]"
              disabled={!allowPaid || suggestAI.isPending}
              onClick={onSuggestAI}
            >
              retry with AI
            </button>
          </div>
        )}
      </div>

      {suggestions.length > 0 && (
        <div className="panel mt-4">
          <div className="panel-h">
            <b>Suggested hard names</b>
            <span className="sub ml-2.5 mb-0">
              capitalized words that recur mid-sentence — likely names. Click to add.
            </span>
          </div>
          <div className="flex flex-wrap gap-2 p-3.5">
            {suggestions.map((s) => (
              <button key={s.term} className="chip" onClick={() => addRow(s.term)} title={s.sample}>
                {s.term} <span className="mono text-ink-3">×{s.count}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
