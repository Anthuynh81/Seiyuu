import { useState } from "react";

import { ApiError } from "../../api/client";
import type { SuggestInput } from "../../api/hooks";
import type { CastAccent, CharacterSummary, SuggestCastResponse, VoiceAssignment, VoiceOut } from "../../api/types";
import { TalkSelect } from "../../components/Select";
import { castRowStatus, manualPicks, proposalMatchesSaved, type CastRowStatus } from "../../lib/casting";

/* -------------------------------------------------- casting */

const KEEP = "__keep__"; // narrator select: send no narrator (server keeps the saved / default)
const SUGGESTED = "__suggested__"; // row select: drop this row's override

const ACCENTS: { value: CastAccent; label: string }[] = [
  { value: "a", label: "American English" },
  { value: "b", label: "British English" },
];

const STATUS_CLASS: Record<CastRowStatus, string> = {
  new: "new",
  "will change": "change",
  "keeps your pick": "kept",
  "your pick": "kept",
  unchanged: "same",
};

function genderLabel(g: string | null): string | null {
  if (!g) return null;
  const k = g.trim().toLowerCase();
  if (k === "f" || k === "female") return "female";
  if (k === "m" || k === "male") return "male";
  if (k === "unknown" || k === "u") return null;
  return k;
}

/** The choices the user makes in the panel; they become the suggest/apply request. */
interface Choices {
  narrator: string;
  accent: CastAccent;
  overrides: Record<string, string>;
}

const START: Choices = { narrator: KEEP, accent: "a", overrides: {} };

function inputFor({ narrator, accent, overrides }: Choices): SuggestInput {
  return { ...(narrator !== KEEP ? { narrator_voice_id: narrator } : {}), accent, overrides };
}

/** The ONE "Suggest a cast" flow: a preview table of the smart caster's proposal (nothing
    written). Every row can be changed — a change becomes an override and re-requests the
    preview, so the table always shows what an apply writes. Narrator and accent feed the same
    request. A change is only kept once the server accepts it: a refused one is rolled back and
    said so, so apply never carries a choice the table doesn't show. Hand picks survive (server
    keep_existing); re-cast is an explicit tick with its re-recording warning; the paid LLM
    caster keeps its 402 confirm. */
export function SuggestCastPanel({
  preview,
  saved,
  characters,
  isMasked,
  voices,
  refreshing,
  unsavedEdits,
  applying,
  applyError,
  onPreview,
  onApply,
  onDismiss,
}: {
  preview: SuggestCastResponse;
  /** the SAVED assignment (null = never cast) — the baseline for "will change" */
  saved: VoiceAssignment | null;
  /** the characters overview, in roster order — the table follows this order */
  characters: CharacterSummary[];
  /** spoiler-safe: a hidden character's row shows no name and no (auto-named) voice */
  isMasked: (characterId: string) => boolean;
  voices: VoiceOut[];
  /** a re-request (row / narrator / accent change) is in flight */
  refreshing: boolean;
  /** local casting edits not yet saved — the post-apply refetch replaces them */
  unsavedEdits: boolean;
  applying: boolean;
  /** the last apply failure — a 402 here grows the explicit paid-confirm recourse */
  applyError: ApiError | null;
  /** re-request the preview; `done` reports the outcome (null = the new preview is shown) */
  onPreview: (input: SuggestInput, done: (error: Error | null) => void) => void;
  onApply: (opts: { input: SuggestInput; recast: boolean; useLlm: boolean; confirmPaid?: boolean }) => void;
  onDismiss: () => void;
}) {
  // `committed` is what the table on screen was computed from (and what apply sends);
  // `pending` is a change the server hasn't answered yet — the selects show it meanwhile.
  const [committed, setCommitted] = useState<Choices>(START);
  const [pending, setPending] = useState<Choices | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [recast, setRecast] = useState(false);
  const [useLlm, setUseLlm] = useState(false);
  const [allowPaid, setAllowPaid] = useState(false);
  const shown = pending ?? committed;

  const requery = (next: Choices) => {
    setPending(next);
    setFailure(null);
    // only the latest request reports back, so an older reply can never commit a stale choice
    onPreview(inputFor(next), (error) => {
      setPending(null);
      if (error) setFailure(error.message);
      else setCommitted(next);
    });
  };

  const proposed = preview.assignment;
  const voiceById = new Map(voices.map((v) => [v.voice_id, v]));
  const charById = new Map(characters.map((c) => [c.id, c]));
  // spoiler-safe: an auto voice is named after its character, so a hidden character's auto
  // voice is left out of every list (the narrator's and the other rows' alike)
  const hiddenVoices = new Set(
    [...characters.map((c) => c.id), ...Object.keys(proposed.assignments)]
      .filter((id) => isMasked(id))
      .map((id) => `${id}_auto`),
  );
  const voiceOptions = voices
    .filter((v) => !hiddenVoices.has(v.voice_id))
    .map((v) => ({ value: v.voice_id, label: `${v.name} · ${v.engine}` }));
  const listed = new Set(voiceOptions.map((o) => o.value));

  const needsPaidConfirm = applyError?.code === "payment_confirmation_required";
  const create = preview.would_create_voice_ids.length;
  const existing = preview.would_recast_voice_ids.length;
  const ctx = {
    saved,
    wouldCreate: preview.would_create_voice_ids,
    wouldRecast: preview.would_recast_voice_ids,
    recast,
  };
  // the roster's order, not the reply's key order (the server appends overridden characters,
  // so a row the user just changed would otherwise jump to the bottom)
  const rank = new Map(characters.map((c, i) => [c.id, i]));
  const rows = Object.entries(proposed.assignments)
    .sort(([a], [b]) => (rank.get(a) ?? characters.length) - (rank.get(b) ?? characters.length))
    .map(([id, voiceId]) => ({ id, voiceId, status: castRowStatus(id, voiceId, ctx) }));
  const changing = rows.filter((r) => r.status === "will change").length;
  const alreadyCast = saved ? rows.filter((r) => saved.assignments[r.id] !== undefined).length : 0;
  // hand picks survive the apply (server keep_existing default) — say so, so nobody fears a clobber
  const kept = manualPicks(proposed.assignments).length;
  const noop = proposalMatchesSaved(proposed, saved, { wouldCreate: preview.would_create_voice_ids, recast });
  const stale = refreshing || pending !== null;
  // Distinctness only holds when no existing auto voice is kept as-is: without re-cast the
  // server skips voices that already exist (often older hash blends it never compares against),
  // so a new character can land close to one of them.
  const distinct = recast || existing === 0;

  const savedNarratorId = saved?.narrator_voice_id;
  const savedNarrator =
    savedNarratorId === undefined
      ? null
      : hiddenVoices.has(savedNarratorId)
        ? "the current narrator"
        : (voiceById.get(savedNarratorId)?.name ?? "the current narrator");
  const narratorOptions = [
    { value: KEEP, label: savedNarrator ? `keep ${savedNarrator}` : "default narrator" },
    ...voiceOptions,
  ];

  const setRow = (charId: string, picked: string, suggested: string) => {
    if (picked === suggested) return; // re-selecting what is already proposed changes nothing
    const overrides = { ...shown.overrides };
    if (picked === SUGGESTED) delete overrides[charId];
    else overrides[charId] = picked;
    requery({ ...shown, overrides });
  };

  return (
    <div className="castprev" role="region" aria-label="suggested cast">
      <div className="caststrip flex-wrap gap-2">
        <span className="tag">suggested cast</span>
        <span className="mono text-[11px] text-ink-2">
          {distinct ? "every character a distinct voice" : "new characters get their own voices"} · {create} new
          {alreadyCast > 0 ? ` · ${alreadyCast} already cast` : ""}
          {changing > 0 ? ` · ${changing} will change` : ""}
        </span>
        {kept > 0 && (
          <span className="mono text-[11px] text-ok">
            Your {kept} pick{kept === 1 ? " is" : "s are"} kept
          </span>
        )}
      </div>
      <div className="caststrip flex-wrap gap-2">
        <span className="mono text-[11px] text-ink-2">narrator</span>
        <TalkSelect
          className="vpick"
          ariaLabel="narrator for the suggested cast"
          value={shown.narrator}
          onChange={(v) => requery({ ...shown, narrator: v })}
          options={narratorOptions}
        />
        <span
          className="mono text-[11px] text-ink-2"
          title="the accent of the voices Seiyuu makes for this cast — voices you picked keep theirs"
        >
          accent
        </span>
        <TalkSelect
          className="vpick"
          ariaLabel="accent for new voices"
          value={shown.accent}
          onChange={(v) => requery({ ...shown, accent: v as CastAccent })}
          options={ACCENTS}
        />
      </div>
      {/* always mounted, so screen readers hear "updating" and a refused change */}
      <div role="status" aria-live="polite" className="mono text-[11px]">
        {stale ? (
          <p className="caststrip text-ink-3">updating the suggestion…</p>
        ) : failure ? (
          <p className="caststrip text-caution">
            couldn't use that change, so it was undone — the table shows the last suggestion that worked. {failure}
          </p>
        ) : null}
      </div>
      <table aria-label="proposed voices">
        <tbody>
          <tr>
            <th>character</th>
            <th>voice</th>
            <th>on apply</th>
          </tr>
          {rows.map(({ id, voiceId, status }) => {
            const c = charById.get(id);
            const name = c?.name ?? id;
            if (isMasked(id)) {
              return (
                <tr key={id} className="masked">
                  <td>
                    <span className="mask">{"▮".repeat(Math.min(Math.max(name.length, 5), 14))}</span>
                    <span className="sample">a character you haven't met yet</span>
                  </td>
                  <td className="vcell">
                    {/* auto voices are named after their characters — showing it would leak the name */}
                    <span className="mono text-ink-3">▮▮</span>
                  </td>
                  <td>
                    <span className={`castst ${STATUS_CLASS[status]}`}>{status}</span>
                  </td>
                </tr>
              );
            }
            const gender = genderLabel(c?.gender ?? null);
            const overridden = id in shown.overrides;
            const options = [
              ...(listed.has(voiceId)
                ? []
                : [
                    {
                      value: voiceId,
                      label: voiceById.has(voiceId) ? "a hidden character's voice" : `new voice for ${name}`,
                    },
                  ]),
              ...(overridden ? [{ value: SUGGESTED, label: "↺ Seiyuu's suggestion" }] : []),
              ...voiceOptions,
            ];
            return (
              <tr key={id}>
                <td>
                  {name}
                  {(gender || c?.age_hint) && (
                    <span className="mt-0.5 flex gap-1">
                      {gender && <span className="castchip">{gender}</span>}
                      {c?.age_hint && <span className="castchip">{c.age_hint}</span>}
                    </span>
                  )}
                </td>
                <td className="vcell">
                  <TalkSelect
                    className="vpick"
                    ariaLabel={`voice for ${name}`}
                    value={voiceId}
                    onChange={(v) => setRow(id, v, voiceId)}
                    options={options}
                  />
                </td>
                <td>
                  <span className={`castst ${STATUS_CLASS[status]}`}>{status}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="caststrip flex-wrap gap-2">
        {existing > 0 && (
          <label className="mono flex items-center gap-[5px] text-[11px]">
            <input type="checkbox" checked={recast} onChange={(e) => setRecast(e.target.checked)} />
            re-cast the {existing} existing (re-renders their audio)
          </label>
        )}
        {!distinct && (
          <span className="mono w-full text-[11px] text-ink-3">
            existing voices are kept as they are and weren't chosen alongside the new ones, so two may sound alike —
            tick re-cast to make every voice distinct
          </span>
        )}
        <label
          className="mono flex items-center gap-[5px] text-[11px]"
          title="ask the LLM for per-character voice-trait hints. Voices stay distinct — the hint only nudges which one each character gets."
        >
          <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
          ✨ use AI trait hints
        </label>
        {unsavedEdits && (
          <span className="mono w-full text-[11px] text-caution">
            your unsaved casting edits are replaced when you apply
          </span>
        )}
        <span className="flex-1" />
        <button className="key quiet px-[9px] py-[3px]" onClick={onDismiss}>
          dismiss
        </button>
        <button
          className="key px-3 py-[3px]"
          disabled={applying || noop || stale}
          title={noop && !stale ? "this is already your cast — change a row, or tick re-cast to rebuild voices" : undefined}
          onClick={() => onApply({ input: inputFor(committed), recast, useLlm })}
        >
          {applying ? "applying…" : noop ? "nothing to apply" : "apply cast"}
        </button>
        {needsPaidConfirm && (
          <>
            <label className="mono flex items-center gap-[5px] text-[11px]">
              <input type="checkbox" checked={allowPaid} onChange={(e) => setAllowPaid(e.target.checked)} />
              approve the paid (Anthropic) caster
            </label>
            <button
              className="key quiet px-[9px] py-[3px]"
              disabled={!allowPaid || applying || stale}
              onClick={() => onApply({ input: inputFor(committed), recast, useLlm, confirmPaid: true })}
            >
              retry cast
            </button>
          </>
        )}
      </div>
    </div>
  );
}
