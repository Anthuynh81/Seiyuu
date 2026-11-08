import { useEffect, useState } from "react";

import {
  useCastPlan,
  useCastTake,
  useDescribeCast,
  useDesign,
  useDesignCast,
  useLiveJobs,
  useUpdateCastPlan,
} from "../../api/hooks";
import type { CastPlan, JobOut, PlanRow, RowUpdate } from "../../api/types";
import { NARRATOR_ROW } from "../../api/types";
import { PlainError } from "../../components/PlainError";
import { takeVerdict } from "../voices/helpers";
import { designMinutes, planRows, rowFacts, rowName, rowStatus, rowsToDesign, shareTargets } from "./castPlan";

const NO_SHARE = "__own__";

/** Qwen-first casting in the Review screen: describe the cast (local LLM), edit the plan,
    design every voice (Qwen3-TTS, GPU), then listen and swap takes per character. */
export function CastPlanPanel({ bookId, onClose }: { bookId: string; onClose: () => void }) {
  const plan = useCastPlan(bookId);
  const live = useLiveJobs();
  const describe = useDescribeCast(bookId);
  const design = useDesignCast(bookId);
  const job = live.data?.jobs.find(
    (j) => j.book_id === bookId && (j.kind === "describe_cast" || j.kind === "design_cast"),
  );

  return (
    <div className="paper mt-2 p-3" role="region" aria-label="design voices">
      <div className="mb-2 flex items-center gap-2">
        <b>Design voices (Qwen)</b>
        <span className="text-xs text-ink-3">a voice made for every character from how the book describes them</span>
        <button className="key quiet ml-auto px-2 py-[2px]" onClick={onClose}>
          close
        </button>
      </div>
      {job ? (
        <JobLine job={job} />
      ) : plan.data ? (
        <PlanTable
          bookId={bookId}
          plan={plan.data}
          onRedescribe={() => describe.mutate()}
          onDesign={() => design.mutate()}
          busy={describe.isPending || design.isPending}
        />
      ) : plan.isLoading ? (
        <p className="m-0 text-ink-3">Loading…</p>
      ) : (
        <div>
          <p className="m-0 text-ink-2">
            First, Seiyuu's local AI reads each character's lines and writes how they should sound. You can change
            every description before any voice is made. Free; takes a few minutes.
          </p>
          <button className="key mt-2" disabled={describe.isPending} onClick={() => describe.mutate()}>
            {describe.isPending ? "starting…" : "Describe the cast"}
          </button>
        </div>
      )}
      {describe.error && <PlainError className="mt-2" error={describe.error} title="couldn't describe the cast" />}
      {design.error && <PlainError className="mt-2" error={design.error} title="couldn't start designing" />}
      {plan.error && <PlainError className="mt-2" error={plan.error} title="couldn't load the cast plan" />}
    </div>
  );
}

function JobLine({ job }: { job: JobOut }) {
  const what = job.kind === "describe_cast" ? "Describing the cast" : "Designing the voices";
  return (
    <p role="status" className="m-0 text-ink-2">
      {what}
      {job.state === "queued" ? " — waiting its turn" : job.progress_text ? `: ${job.progress_text}` : "…"}
    </p>
  );
}

function PlanTable({
  bookId,
  plan,
  onRedescribe,
  onDesign,
  busy,
}: {
  bookId: string;
  plan: CastPlan;
  onRedescribe: () => void;
  onDesign: () => void;
  busy: boolean;
}) {
  const update = useUpdateCastPlan(bookId);
  const save = (u: RowUpdate) => update.mutate([u]);
  const todo = rowsToDesign(plan);
  const protagonist = plan.protagonist_suggestion;
  return (
    <>
      {plan.first_person && protagonist && plan.narrator.same_as === null && (
        <p className="m-0 mb-2 text-xs text-ink-2">
          This book is told in the first person.{" "}
          <button
            className="link"
            onClick={() => save({ character_id: NARRATOR_ROW, same_as: protagonist })}
          >
            Narrate in {rowName(plan, protagonist)}'s voice
          </button>
        </p>
      )}
      <ul className="m-0 list-none p-0" aria-label="cast plan">
        {planRows(plan).map((row) => (
          <PlanRowView key={row.character_id} bookId={bookId} plan={plan} row={row} onSave={save} />
        ))}
      </ul>
      {plan.notes.map((n) => (
        <p key={n} className="m-0 mt-1 text-xs text-clip">
          {n}
        </p>
      ))}
      {update.error && <PlainError className="mt-2" error={update.error} title="couldn't save that change" />}
      <div className="mt-3 flex items-center gap-2">
        {todo.length > 0 ? (
          <button className="key" disabled={busy || update.isPending} onClick={onDesign}>
            Design {todo.length} voice{todo.length === 1 ? "" : "s"} (about {designMinutes(todo)} min)
          </button>
        ) : (
          <span className="text-xs text-ink-2">Every voice is designed and cast.</span>
        )}
        <button className="key quiet" disabled={busy} onClick={onRedescribe}>
          describe again
        </button>
        <span className="text-[11px] text-ink-3">your edited rows are kept</span>
      </div>
    </>
  );
}

function PlanRowView({
  bookId,
  plan,
  row,
  onSave,
}: {
  bookId: string;
  plan: CastPlan;
  row: PlanRow;
  onSave: (u: RowUpdate) => void;
}) {
  const id = row.character_id;
  const isNarrator = id === NARRATOR_ROW;
  const [draft, setDraft] = useState(row.description);
  useEffect(() => setDraft(row.description), [row.description]);
  const [open, setOpen] = useState(false);
  const shares = row.same_as !== null;
  return (
    <li className="border-b border-hairline py-2 last:border-b-0" aria-label={row.name}>
      <div className="flex flex-wrap items-center gap-2">
        <b>{row.name}</b>
        <span className="text-[11px] text-ink-3">{rowFacts(row)}</span>
        <span className="ml-auto text-[11px] text-ink-2">{rowStatus(plan, row)}</span>
      </div>
      {!shares && !row.keep_current && (
        <>
          <label className="sr-only" htmlFor={`desc-${id}`}>
            how {row.name} sounds
          </label>
          <textarea
            id={`desc-${id}`}
            rows={2}
            className="mt-1 w-full"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => {
              const text = draft.trim();
              if (text.length >= 3 && text !== row.description) onSave({ character_id: id, description: text });
            }}
          />
          {row.sample_lines.length > 0 && (
            <select
              aria-label={`${row.name}'s sample line`}
              className="mt-1 w-full text-xs"
              value={row.sample_index}
              onChange={(e) => onSave({ character_id: id, sample_index: Number(e.target.value) })}
            >
              {row.sample_lines.map((line, i) => (
                <option key={i} value={i}>
                  {line.length > 110 ? `${line.slice(0, 110)}…` : line}
                </option>
              ))}
            </select>
          )}
        </>
      )}
      <div className="mt-1 flex flex-wrap items-center gap-3 text-xs">
        {!isNarrator && (
          <label className="flex items-center gap-1">
            same voice as
            <select
              aria-label={`${row.name} shares a voice with`}
              value={row.same_as ?? NO_SHARE}
              onChange={(e) =>
                onSave(
                  e.target.value === NO_SHARE
                    ? { character_id: id, clear_same_as: true }
                    : { character_id: id, same_as: e.target.value },
                )
              }
            >
              <option value={NO_SHARE}>— its own voice —</option>
              {shareTargets(plan, row).map((t) => (
                <option key={t.character_id} value={t.character_id}>
                  {t.name}
                </option>
              ))}
            </select>
          </label>
        )}
        {isNarrator && row.same_as !== null && (
          <button className="link" onClick={() => onSave({ character_id: id, clear_same_as: true })}>
            give the narrator its own voice
          </button>
        )}
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={row.keep_current}
            onChange={(e) => onSave({ character_id: id, keep_current: e.target.checked })}
          />
          keep current voice
        </label>
        {row.design_id !== null && row.voice_id !== null && !shares && (
          <button className="link ml-auto" onClick={() => setOpen(!open)}>
            {open ? "hide takes" : "hear takes"}
          </button>
        )}
      </div>
      {open && row.design_id && <Takes bookId={bookId} row={row} />}
    </li>
  );
}

function Takes({ bookId, row }: { bookId: string; row: PlanRow }) {
  const design = useDesign(row.design_id, false);
  const swap = useCastTake(bookId);
  if (!design.data) return <p className="m-0 mt-1 text-xs text-ink-3">Loading takes…</p>;
  return (
    <ul className="m-0 mt-1 list-none p-0" aria-label={`${row.name}'s takes`}>
      {design.data.takes.map((t) => {
        const current = t.kept_voice_id !== null && t.kept_voice_id === row.voice_id;
        return (
          <li key={t.take} className="py-1" aria-label={`take ${t.take}`}>
            <div className="flex items-center gap-2 text-xs">
              <b>take {t.take}</b>
              <span className="text-ink-3">{takeVerdict(t)}</span>
              {current ? (
                <span className="ml-auto text-ink-2">in use</span>
              ) : (
                <button
                  className="key quiet ml-auto px-2 py-[2px]"
                  disabled={swap.isPending}
                  onClick={() => swap.mutate({ characterId: row.character_id, take: t.take })}
                >
                  use this take
                </button>
              )}
            </div>
            <audio controls preload="none" src={t.audio_url} className="mt-1 h-[28px] w-full" />
          </li>
        );
      })}
      {swap.error && <PlainError className="mt-1" error={swap.error} title="couldn't switch takes" />}
    </ul>
  );
}
