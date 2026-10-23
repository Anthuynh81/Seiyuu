import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  useAssignment,
  useBooks,
  useLexicon,
  usePreviewLexicon,
  useSaveLexicon,
  useSuggestRespellings,
  useVoices,
} from "../api/hooks";
import type { LexiconEntry, SuggestedTerm, VoiceOut } from "../api/types";
import { useScreenBook, withBook } from "../app/currentBook";
import { NextStep } from "../components/NextStep";
import { PlainError } from "../components/PlainError";
import { TalkSelect } from "../components/Select";
import { Tip } from "../components/Tooltip";
import { explainFailure, failureOf, plural, SCREENS, term as uiTerm } from "../lib/glossary";
import { applyRespellings, blankEntry, cleanForSave, duplicateTerms, entriesSig, hearItLine } from "../lib/lexicon";
import { ADD_A_BOOK } from "../lib/nextStep";
import { PlayableRow } from "./review/PlayableRow";

const sameTerm = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();

/* A per-book pronunciation dictionary: word -> how to say it (spoken on every engine), with an
   optional phonetic spelling (IPA, under Advanced) that only the Kokoro engine uses. Saving
   re-records only the lines whose text changes; the count is shown before and after a save.
   Deep link: ?book=<id>&term=<word> (Listen's "Fix pronunciation", audio checks) opens that
   book with the word waiting in a new row. */

const HEAR_TIP =
  "▶ plays a sentence with your respelling in the narrator's voice. It's approximate: the recording applies the word the same way, but the voice may read the sentence a little differently.";

const UNCAST_REASON = "cast the voices first to hear it in the narrator's voice";

function EntryRow({
  entry,
  onChange,
  onRemove,
  duplicate,
  advanced,
  narrator,
  sample,
}: {
  entry: LexiconEntry;
  onChange: (next: LexiconEntry) => void;
  onRemove: () => void;
  duplicate: boolean;
  advanced: boolean;
  narrator: VoiceOut | undefined;
  sample: string | undefined;
}) {
  const set = (patch: Partial<LexiconEntry>) => onChange({ ...entry, ...patch });
  const line = hearItLine(entry, sample);
  const word = entry.term.trim();
  return (
    <PlayableRow
      colSpan={advanced ? 6 : 5}
      cells={
        <td>
          <input
            className={`taginput ${duplicate ? "border-clip" : ""}`}
            value={entry.term}
            placeholder="as written"
            aria-label="word"
            onChange={(e) => set({ term: e.target.value })}
          />
        </td>
      }
      picker={
        <input
          className="taginput w-full"
          value={entry.respelling}
          placeholder="say it like this"
          aria-label="say it like"
          onChange={(e) => set({ respelling: e.target.value })}
        />
      }
      // the ▶ speaks a sentence with the respelling swapped in, in the narrator's voice
      voice={line ? narrator : undefined}
      text={line}
      who={`“${word}” in a sentence`}
      unavailable={!narrator ? UNCAST_REASON : !line ? "type how to say it first" : undefined}
      after={
        <>
          {advanced && (
            <td>
              <input
                className="taginput"
                value={entry.ipa ?? ""}
                placeholder="IPA (Kokoro only)"
                aria-label="phonetic spelling (IPA)"
                onChange={(e) => set({ ipa: e.target.value || null })}
              />
            </td>
          )}
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
              aria-label="match capitals exactly"
              title="only change the word when its capital letters match exactly"
              onChange={(e) => set({ case_sensitive: e.target.checked })}
            />
          </td>
          <td>
            <button className="key quiet" onClick={onRemove} aria-label="remove">
              ✕
            </button>
          </td>
        </>
      }
    />
  );
}

export function Lexicon() {
  const [params, setParams] = useSearchParams();
  // ?term=: arrived from Listen's "Fix pronunciation" — that word waits in a new row
  const wantTerm = (params.get("term") ?? "").trim();
  const books = useBooks();
  const ingested = books.data?.books.filter((b) => b.ingested) ?? [];
  // no ?book=: the book this viewer last worked on (if imported), else the first imported one
  const bookId = useScreenBook(books.data ? ingested.map((b) => b.book_id) : undefined, ingested[0]?.book_id);

  const card = ingested.find((b) => b.book_id === bookId);

  const lexicon = useLexicon(bookId);
  // "hear it in a sentence" speaks in the book's narrator voice — known once the book is cast
  const assignment = useAssignment(bookId, !!card?.assigned);
  const voices = useVoices({ enabled: !!assignment.data });
  const narrator = voices.data?.voices.find((v) => v.voice_id === assignment.data?.narrator_voice_id);
  const save = useSaveLexicon(bookId ?? "");
  const preview = usePreviewLexicon(bookId ?? "");
  const suggestAI = useSuggestRespellings(bookId ?? "");

  const [rows, setRows] = useState<LexiconEntry[]>([]);
  const [savedInfo, setSavedInfo] = useState<{ affected: number; total: number } | null>(null);
  const [allowPaid, setAllowPaid] = useState(false);
  // Advanced (the IPA column): open by default only when an entry already uses it. The
  // viewer's choice belongs to ONE book: switching books goes back to that default, so a
  // book whose entries carry IPA never opens with it hidden.
  const [advancedChoice, setAdvancedChoice] = useState<{ book: string | undefined; open: boolean } | null>(null);

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

  // Add the ?term= word as a new, unsaved row once the book's entries are in (once per book and
  // term; a word already in the dictionary is left as it is). Runs after the seeding effect
  // above, so its updater sees the seeded entries. The row goes FIRST and its respelling box
  // takes focus: in a long dictionary a row appended at the bottom is hard to find.
  const termAdded = useRef<string | null>(null);
  const tableRef = useRef<HTMLTableElement>(null);
  const [focusTermRow, setFocusTermRow] = useState(0);
  useEffect(() => {
    if (!wantTerm || !bookId || !lexicon.data) return;
    const key = `${bookId}|${wantTerm}`;
    if (termAdded.current === key) return;
    termAdded.current = key;
    const have = lexicon.data.entries.some((r) => sameTerm(r.term, wantTerm));
    setRows((rs) => (rs.some((r) => sameTerm(r.term, wantTerm)) ? rs : [blankEntry(wantTerm), ...rs]));
    if (!have) setFocusTermRow((n) => n + 1);
  }, [bookId, wantTerm, lexicon.data]);
  useEffect(() => {
    if (focusTermRow) tableRef.current?.querySelector<HTMLInputElement>('input[aria-label="say it like"]')?.focus();
  }, [focusTermRow]);
  // the hint asks for a respelling only while the word's row is still waiting for one
  const termWaiting = !!wantTerm && rows.some((r) => sameTerm(r.term, wantTerm) && !r.respelling.trim());

  const dupes = duplicateTerms(rows);
  const dirty = serverSig !== null && entriesSig(rows) !== serverSig;
  const canSave = dirty && dupes.length === 0 && !save.isPending;

  const advanced =
    (advancedChoice?.book === bookId ? advancedChoice.open : null) ?? rows.some((r) => !!r.ipa?.trim());
  // the book's own sentence for a word, when the hard-name finder saw it
  const sampleFor = (word: string) =>
    lexicon.data?.suggestions.find((s) => sameTerm(s.term, word))?.sample;

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
        <h1>{SCREENS.lexicon}</h1>
        <p className="sub">No imported books yet. Add one in the {SCREENS.library} first.</p>
        <NextStep step={ADD_A_BOOK} />
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
  const changed = (n: number, total: number) => `${n} of ${total} ${plural("block", total)}`;

  return (
    <section className="screen">
      <h1 className="flex items-baseline gap-3.5">
        {SCREENS.lexicon}
        <TalkSelect
          className="bookpick"
          ariaLabel="book"
          value={bookId}
          onChange={(v) => setParams({ book: v })}
          options={ingested.map((b) => ({ value: b.book_id, label: b.title ?? b.book_id }))}
        />
      </h1>
      <p className="sub">
        Fix mispronounced names and invented words for this book: type each word as it is written
        and how to say it. Every voice uses it. Saving re-records only the{" "}
        {plural("segment", 2)} that contain a changed word, the next time you record.
      </p>
      {termWaiting && (
        <p className="mono -mt-1 mb-3 text-[11px] text-ink-2">
          Type how “{wantTerm}” should sound in the first row below, then save.
        </p>
      )}

      {lexicon.isError && <PlainError error={lexicon.error} title="couldn't load the dictionary" />}

      <div className="panel m-0">
        <div className="panel-h">
          <b>Words</b>
          <span className="tag ml-2">{cleanForSave(rows).length}</span>
        </div>
        <div className="overflow-x-auto px-3.5 pt-2">
          <table ref={tableRef} className="w-full border-collapse">
            <thead>
              <tr className="mono text-left text-[11px] text-ink-3">
                <th className="px-1.5 py-1 font-semibold">word</th>
                <th className="px-1.5 py-1 font-semibold">
                  <Tip content={HEAR_TIP}>
                    <span tabIndex={0}>say it like · ▶ approximate</span>
                  </Tip>
                </th>
                {advanced && <th className="px-1.5 py-1 font-semibold">IPA (Kokoro only)</th>}
                <th className="px-1.5 py-1 font-semibold">note</th>
                <th className="px-1.5 py-1 font-semibold">{uiTerm("caseSensitive")}</th>
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
                  advanced={advanced}
                  narrator={narrator}
                  sample={sampleFor(e.term)}
                />
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={advanced ? 6 : 5} className="sub px-1.5 py-2.5">
                    no words yet — add one below or pick a suggested name
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2.5 px-3.5 pb-3.5">
          <button className="key quiet" onClick={() => addRow()}>
            + add a word
          </button>
          <button
            className="key quiet"
            disabled={!dirty || dupes.length > 0 || preview.isPending}
            onClick={() => preview.mutate(cleanForSave(rows))}
            title={`how many ${plural("block", 2)} this change would re-record`}
          >
            preview impact
          </button>
          <button className="key" disabled={!canSave} onClick={onSave}>
            {save.isPending ? "saving…" : "save"}
          </button>

          <span className="flex-1" />
          <button
            className="key quiet"
            aria-expanded={advanced}
            onClick={() => setAdvancedChoice({ book: bookId, open: !advanced })}
            title="a phonetic spelling (IPA) for words a respelling can't capture — only the Kokoro voices use it"
          >
            {advanced ? "hide advanced" : "advanced: IPA"}
          </button>
          <button
            className="key quiet"
            disabled={suggestAI.isPending}
            onClick={onSuggestAI}
            title="ask the AI to suggest how to say the words above, or (if there are none) the book's likely hard names. Only suggestions: you still review and save."
          >
            {suggestAI.isPending ? "asking AI…" : "✨ suggest with AI"}
          </button>

          {dupes.length > 0 && (
            <span className="mono text-[11px] text-clip">listed twice: {dupes.join(", ")}</span>
          )}
          {preview.data && !preview.isPending && (
            <span className="mono text-[11px] text-ink-2">
              would re-record {changed(preview.data.affected_blocks, preview.data.total_speakable_blocks)}
            </span>
          )}
          {savedInfo && !dirty && (
            <span className="mono text-[11px] text-ink-2">
              saved · {changed(savedInfo.affected, savedInfo.total)} will be re-recorded the next time you record
              {savedInfo.affected > 0 && card?.rendered && (
                <>
                  {" · "}
                  <Link className="link" to={withBook("/render", bookId)}>
                    re-record the changed chapters on {SCREENS.render} →
                  </Link>
                </>
              )}
            </span>
          )}
          {suggestAI.data && !suggestAI.isPending && (
            <Tip content={`${suggestAI.data.provider} · ${suggestAI.data.model}`}>
              <span className="mono text-[11px] text-ink-2" tabIndex={0}>
                {suggestAI.data.suggestions.length} suggestion
                {suggestAI.data.suggestions.length === 1 ? "" : "s"} from the AI — review and save
              </span>
            </Tip>
          )}
        </div>
        {save.isError && <PlainError className="mx-3.5 mb-3.5" error={save.error} title="couldn't save" />}
        {aiError && !needsPaidConfirm && (
          <PlainError className="mx-3.5 mb-3.5" error={aiError} title="couldn't suggest" />
        )}

        {needsPaidConfirm && (
          <div className="mt-2 flex flex-wrap items-center gap-2 px-3.5 pb-3.5">
            <p className="w-full text-[12.5px] text-ink" title={aiError?.message}>
              {explainFailure(failureOf(aiError)).summary} Nothing has been spent yet.
            </p>
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
