import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useBooks, useLexicon, usePreviewLexicon, useSaveLexicon } from "../api/hooks";
import type { LexiconEntry, SuggestedTerm } from "../api/types";
import { blankEntry, cleanForSave, duplicateTerms, entriesSig } from "../lib/lexicon";

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
          className="taginput"
          value={entry.term}
          placeholder="as written"
          aria-label="term"
          onChange={(e) => set({ term: e.target.value })}
          style={duplicate ? { borderColor: "var(--clip)" } : undefined}
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
      <td style={{ textAlign: "center" }}>
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

  const [rows, setRows] = useState<LexiconEntry[]>([]);
  const [savedInfo, setSavedInfo] = useState<{ affected: number; total: number } | null>(null);

  // Re-seed the editable rows whenever the server copy (or the selected book) changes.
  const serverSig = lexicon.data ? entriesSig(lexicon.data.entries) : null;
  useEffect(() => {
    if (lexicon.data) setRows(lexicon.data.entries.map((e) => ({ ...e })));
    setSavedInfo(null);
    preview.reset();
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
      onSuccess: (res) =>
        setSavedInfo({ affected: res.affected_blocks, total: res.total_speakable_blocks }),
    });
  };

  return (
    <section className="screen">
      <h1 style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
        Pronunciation
        <select
          className="bookpick"
          value={bookId}
          onChange={(e) => setParams({ book: e.target.value })}
          aria-label="book"
        >
          {ingested.map((b) => (
            <option key={b.book_id} value={b.book_id}>
              {b.title ?? b.book_id}
            </option>
          ))}
        </select>
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

      <div className="panel" style={{ margin: 0 }}>
        <div className="panel-h">
          <b>Entries</b>
          <span className="tag" style={{ marginLeft: 8 }}>
            {cleanForSave(rows).length}
          </span>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr className="mono" style={{ textAlign: "left", color: "var(--ink-3)", fontSize: 11 }}>
                <th style={{ padding: "4px 6px" }}>term</th>
                <th style={{ padding: "4px 6px" }}>respelling</th>
                <th style={{ padding: "4px 6px" }}>ipa (kokoro)</th>
                <th style={{ padding: "4px 6px" }}>note</th>
                <th style={{ padding: "4px 6px" }}>case</th>
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
                  <td colSpan={6} className="sub" style={{ padding: "10px 6px" }}>
                    no entries yet — add a term below or pick from suggestions
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="row" style={{ gap: 10, marginTop: 12, alignItems: "center", flexWrap: "wrap" }}>
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

          {dupes.length > 0 && (
            <span className="mono" style={{ color: "var(--clip)", fontSize: 11 }}>
              duplicate term: {dupes.join(", ")}
            </span>
          )}
          {preview.data && !preview.isPending && (
            <span className="mono" style={{ color: "var(--ink-2)", fontSize: 11 }}>
              would re-synthesize {preview.data.affected_blocks} of {preview.data.total_speakable_blocks} segments
            </span>
          )}
          {savedInfo && !dirty && (
            <span className="mono" style={{ color: "var(--ink-2)", fontSize: 11 }}>
              saved · {savedInfo.affected} of {savedInfo.total} segments will re-synthesize on next render
            </span>
          )}
          {save.isError && (
            <span className="mono" style={{ color: "var(--clip)", fontSize: 11 }}>
              {(save.error as Error).message}
            </span>
          )}
        </div>
      </div>

      {suggestions.length > 0 && (
        <div className="panel" style={{ marginTop: 16 }}>
          <div className="panel-h">
            <b>Suggested hard names</b>
            <span className="sub" style={{ marginLeft: 10 }}>
              capitalized words that recur mid-sentence — likely names. Click to add.
            </span>
          </div>
          <div className="row" style={{ gap: 8, flexWrap: "wrap", marginTop: 8 }}>
            {suggestions.map((s) => (
              <button key={s.term} className="chip" onClick={() => addRow(s.term)} title={s.sample}>
                {s.term} <span className="mono" style={{ color: "var(--ink-3)" }}>×{s.count}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
