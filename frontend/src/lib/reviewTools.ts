import type { CharacterSummary, SegmentRow } from "../api/types";
import { chapterOfBlock } from "../api/types";
import { genderLabel } from "./casting";

/** Character Review power tools: pure helpers that turn server data (the edit log, edit
    warnings, registry notes, character hints) into plain words and line links. No I/O. */

/** A line on the page a "Go to line" can open (chapter from the paragraph id). */
export interface LineTarget {
  chapter: number;
  block: string;
  /** the line within the paragraph; null = the paragraph's first line */
  seg: number | null;
}

function targetOf(block: string, seg: number | null): LineTarget | null {
  const chapter = chapterOfBlock(block);
  return chapter === null ? null : { chapter, block, seg };
}

/* -------------------------------------------------- confirmed lines */

/** A line someone already set or confirmed by hand: a manual reassign is ground truth, so the
    server gives it confidence 1.0 (services/edits.py). */
export const isConfirmed = (row: SegmentRow) => row.confidence >= 1;

/** A line has audio the shared player can play: a rendered wav and the timing to show it. */
export const canPlay = (row: SegmentRow) =>
  row.has_audio && row.audio_segment !== null && row.duration_seconds !== null;

/* -------------------------------------------------- gender / age */

/** The voice family automatic casting picks: only male-ish hints get a male family, anything
    else (female, unknown, or another word) gets a female one — voices/casting.py `_family`. */
const MALE_HINTS = new Set(["male", "m", "man", "boy"]);
export function castFamily(gender: string | null | undefined): "male" | "female" {
  return MALE_HINTS.has((gender ?? "").trim().toLowerCase()) ? "male" : "female";
}

/** Chips for a character's hints, in words; the gender chip says "gender unknown" when absent. */
export function characterHints(c: Pick<CharacterSummary, "gender" | "age_hint">): {
  gender: string;
  genderKnown: boolean;
  age: string | null;
} {
  const g = genderLabel(c.gender);
  const age = c.age_hint?.trim() || null;
  return { gender: g ?? "gender unknown", genderKnown: g !== null, age };
}

/** Words the caster's `_family` sends to the female family that really do mean female: no
    warning for these, the guess is right. */
const FEMALE_HINTS = new Set(["female", "f", "woman", "girl", "lady"]);

/** The plain warning for a character automatic casting will treat as female without knowing.
    null when there's nothing to warn about (male, or a word that means female). */
export function genderWarning(gender: string | null | undefined): string | null {
  const k = (gender ?? "").trim().toLowerCase();
  if (FEMALE_HINTS.has(k) || castFamily(gender) === "male") return null;
  const g = genderLabel(gender);
  const fix = "If that's wrong, pick a voice for them by hand.";
  return g === null
    ? `Gender unknown, so automatic casting gives them a female voice. ${fix}`
    : `Automatic casting only tells male from female, so "${g}" gets a female voice. ${fix}`;
}

/* -------------------------------------------------- Python repr strings in server notes */

/** The quoted strings in a Python repr fragment (`'Mr. Smith'`, `"O'Neil"`), unescaped. Server
    notes quote names with `!r`, so this is how names are read back out of them. */
export function reprStrings(s: string): string[] {
  const out: string[] = [];
  const re = /'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"/g;
  for (const m of s.matchAll(re)) {
    const raw = m[1] ?? m[2] ?? "";
    out.push(raw.replace(/\\(.)/g, "$1"));
  }
  return out;
}

/* -------------------------------------------------- merge suggestions from registry notes */

export interface MergeSuggestion {
  /** stable per pair: the two character ids, sorted */
  key: string;
  /** the one whose lines move (the less busy one by default) */
  loser: CharacterSummary;
  /** the one that stays */
  winner: CharacterSummary;
  /** why Seiyuu thinks they might be one person, in plain words */
  why: string;
}

interface NoteRule {
  match: RegExp;
  /** groups of names: every pair across the groups is a candidate */
  names: (m: RegExpExecArray) => string[][];
  why: string;
}

// Which registry notes are merge candidates (attribute/aliases.py, attribute/registry.py):
// only the ones that mean "possibly the same person". Left out on purpose, because the server
// already judged the pair to be DIFFERENT people (precision over recall, never over-merge):
//  - "alias: ambiguous '<key>' shared by [...]": written only when the group has a gender,
//    generation or title clash (aliases.py `_conflict`, e.g. Mr. and Mrs. Bennet);
//  - "alias: adjudicator rejected ..." and "alias: vetoed ...".
const NOTE_RULES: NoteRule[] = [
  {
    match: /^alias: adjudicator approved (.*?) at confidence/,
    names: (m) => reprStrings(m[1]).map((n) => [n]),
    why: "The AI thought these are probably the same person, but wasn't sure enough to merge them.",
  },
  {
    match: /^not merging (.*?) with existing \[(.*)\]/,
    names: (m) => [reprStrings(m[1]), reprStrings(m[2])],
    why: "These names matched while the book was read, but Seiyuu didn't merge them on its own.",
  },
];

/* The server's "cannot be the same person" guard (aliases.py `_title_class`, `_norm_gender`,
   `_effective_gender`, `_conflict`), mirrored so a card never proposes a pair it would veto,
   e.g. a "not merging" note naming Mr. and Mrs. Smith. */
const TITLE_CLASS: Record<string, string> = {
  mr: "male_adult", "mr.": "male_adult", mister: "male_adult", sir: "male_adult", lord: "male_adult",
  mrs: "female_adult", "mrs.": "female_adult", lady: "female_adult", madam: "female_adult", mistress: "female_adult",
  miss: "female_unmarried", "miss.": "female_unmarried", ms: "female_unmarried", "ms.": "female_unmarried",
  master: "child", young: "child",
};
const TITLE_GENDER: Record<string, string> = {
  male_adult: "male",
  female_adult: "female",
  female_unmarried: "female",
};
const titleClass = (name: string) => TITLE_CLASS[name.trim().toLowerCase().split(/\s+/)[0] ?? ""] ?? "none";
function effectiveGender(c: Pick<CharacterSummary, "name" | "gender">): string {
  const g = (c.gender ?? "").trim().toLowerCase();
  if (["male", "m", "man", "boy"].includes(g)) return "male";
  if (["female", "f", "woman", "girl"].includes(g)) return "female";
  return TITLE_GENDER[titleClass(c.name)] ?? "unknown";
}
/** True when two characters cannot be one person: a gender clash, or two different titles. */
export function cannotBeSame(
  a: Pick<CharacterSummary, "name" | "gender">,
  b: Pick<CharacterSummary, "name" | "gender">,
): boolean {
  const ga = effectiveGender(a);
  const gb = effectiveGender(b);
  if (ga !== "unknown" && gb !== "unknown" && ga !== gb) return true;
  const ca = titleClass(a.name);
  const cb = titleClass(b.name);
  return ca !== "none" && cb !== "none" && ca !== cb;
}

/** Find the character a note's name means today: the exact name, then case-insensitively,
    then a unique alias. A name merged away since resolves to its new owner. */
function resolveName(name: string, cast: readonly CharacterSummary[]): CharacterSummary | null {
  const exact = cast.find((c) => c.name === name);
  if (exact) return exact;
  const lower = name.toLowerCase();
  const loose = cast.filter((c) => c.name.toLowerCase() === lower);
  if (loose.length === 1) return loose[0];
  const byAlias = cast.filter((c) => c.aliases.some((a) => a.toLowerCase() === lower));
  return byAlias.length === 1 ? byAlias[0] : null;
}

/** Merge cards from the attribution's registry notes. Only pairs that are still two different
    characters, and that could be one person, show up (a merge already made, by hand or by a re-run, drops its card). The less
    busy character's lines move to the busier one by default. */
export function mergeSuggestions(
  notes: readonly string[],
  cast: readonly CharacterSummary[],
): MergeSuggestion[] {
  const out: MergeSuggestion[] = [];
  const seen = new Set<string>();
  for (const note of notes) {
    for (const rule of NOTE_RULES) {
      const m = rule.match.exec(note);
      if (!m) continue;
      const groups = rule.names(m);
      const pairs: [string, string][] = [];
      for (let i = 0; i < groups.length; i++)
        for (let j = i + 1; j < groups.length; j++)
          for (const a of groups[i]) for (const b of groups[j]) pairs.push([a, b]);
      for (const [na, nb] of pairs) {
        const a = resolveName(na, cast);
        const b = resolveName(nb, cast);
        if (!a || !b || a.id === b.id || cannotBeSame(a, b)) continue;
        const key = [a.id, b.id].sort().join("|");
        if (seen.has(key)) continue;
        seen.add(key);
        const [winner, loser] = b.line_count > a.line_count ? [b, a] : [a, b];
        out.push({ key, winner, loser, why: rule.why });
      }
      break;
    }
  }
  return out;
}

export const linesWord = (n: number) => `${n.toLocaleString()} line${n === 1 ? "" : "s"}`;

/** "moves N lines" — what a merge does, in words. */
export function mergePreview(loser: Pick<CharacterSummary, "name" | "line_count">, winnerName: string): string {
  return loser.line_count > 0
    ? `Moves ${linesWord(loser.line_count)} from ${loser.name} to ${winnerName}. "${loser.name}" stays as another name for ${winnerName}.`
    : `${loser.name} has no lines; "${loser.name}" becomes another name for ${winnerName}.`;
}

/* -------------------------------------------------- the edit log, in words */

/** One op from GET /edits (services/edits.py EditOp, anchors filled by the server). */
export type EditOp =
  | { op: "rename"; character_id: string; new_name: string; expected_name?: string | null }
  | {
      op: "merge";
      loser_id: string;
      winner_id: string;
      expected_loser_name?: string | null;
      expected_winner_name?: string | null;
    }
  | { op: "reassign"; block_id: string; segment_index: number; speaker: string | null; text_anchor?: string | null }
  | {
      op: "set_emotion";
      block_id: string;
      segment_index: number;
      emotion: { label: string; intensity: number } | null;
      text_anchor?: string | null;
    };

export interface EditDescription {
  /** what the change did, in plain words */
  text: string;
  /** the line it changed, when it changed one */
  target: LineTarget | null;
}

const INTENSITY = ["", "gently", "clearly", "strongly"];

const quoteOf = (anchor: string | null | undefined) => {
  const a = (anchor ?? "").trim();
  return a ? `"${a}${a.length >= 60 ? "…" : ""}"` : "a line";
};

/** Describe one edit-log entry. `nameOf(id)` returns the name to show for a character id (null
    when the character is gone, e.g. merged away); a masked one comes back already hidden.
    `isHidden(id)` says spoiler-safe hides that character: a rename or merge touching it then
    leaves out the names the log stored (the anchors), which would give it away. */
export function describeEdit(
  raw: Record<string, unknown>,
  nameOf: (id: string) => string | null,
  isHidden: (id: string) => boolean = () => false,
): EditDescription {
  const op = raw as EditOp;
  const who = (id: string, fallback?: string | null) => nameOf(id) ?? fallback ?? "a character no longer in the cast";
  switch (op.op) {
    case "rename":
      if (isHidden(op.character_id)) return { text: "Renamed a hidden character", target: null };
      return {
        text: `Renamed ${op.expected_name ? `"${op.expected_name}"` : who(op.character_id)} to "${op.new_name}"`,
        target: null,
      };
    case "merge": {
      if (isHidden(op.winner_id) || isHidden(op.loser_id))
        return { text: "Merged a name into a hidden character", target: null };
      // the names the person saw when merging (anchors); the loser is gone after the merge
      const loser = op.expected_loser_name ?? who(op.loser_id);
      const winner = nameOf(op.winner_id) ?? op.expected_winner_name ?? who(op.winner_id);
      return { text: `Merged ${loser} into ${winner}`, target: null };
    }
    case "reassign":
      return {
        text:
          op.speaker === null
            ? `Narrator reads ${quoteOf(op.text_anchor)}`
            : `${who(op.speaker)} says ${quoteOf(op.text_anchor)}`,
        target: targetOf(op.block_id, op.segment_index),
      };
    case "set_emotion":
      return {
        text: op.emotion
          ? `Emotion set to ${op.emotion.label}${INTENSITY[op.emotion.intensity] ? ` (${INTENSITY[op.emotion.intensity]})` : ""} on ${quoteOf(op.text_anchor)}`
          : `Emotion cleared on ${quoteOf(op.text_anchor)}`,
        target: targetOf(op.block_id, op.segment_index),
      };
    default:
      return { text: "A change this version of Seiyuu doesn't recognise", target: null };
  }
}

/* -------------------------------------------------- edit warnings, in words */

export interface WarningDescription {
  text: string;
  target: LineTarget | null;
  /** the server's own wording, for a details line */
  raw: string;
}

/** An edit-overlay warning (an earlier fix that no longer applies) in plain words, with the
    line it was about when there is one to open (services/edits.py `_apply_*` messages). */
export function describeEditWarning(w: string): WarningDescription {
  const kind = /^(\w+) skipped:/.exec(w)?.[1];
  const what =
    kind === "reassign"
      ? "A speaker fix"
      : kind === "set_emotion"
        ? "An emotion tag"
        : kind === "rename"
          ? "A rename"
          : kind === "merge"
            ? "A merge"
            : "An earlier fix";
  const at = /'([^']+)'\[(\d+)\]/.exec(w);
  const inBlock = /block '([^']+)' has \d+ segment\(s\), index (\d+)/.exec(w);
  if (/is now different text/.test(w) && at) {
    return {
      text: `${what} no longer applies: the line changed since you made it. Make it again if you still want it.`,
      target: targetOf(at[1], Number(at[2])),
      raw: w,
    };
  }
  if (inBlock) {
    return {
      text: `${what} no longer applies: that line isn't in the paragraph any more. Check the paragraph and make it again if you still want it.`,
      target: targetOf(inBlock[1], null),
      raw: w,
    };
  }
  if (/not in this attribution/.test(w) && /block '/.test(w)) {
    return { text: `${what} no longer applies: that paragraph is no longer in the book.`, target: null, raw: w };
  }
  if (/no character .* in this attribution|not distinct\/present/.test(w)) {
    return {
      text: `${what} no longer applies: a character it names is no longer in the cast.`,
      target: null,
      raw: w,
    };
  }
  if (/no longer denote|is now '/.test(w)) {
    return {
      text: `${what} no longer applies: the characters changed since you made it. Make it again if you still want it.`,
      target: null,
      raw: w,
    };
  }
  return { text: `${what} no longer applies.`, target: null, raw: w };
}
