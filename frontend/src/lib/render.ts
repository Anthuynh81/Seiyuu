import type { ChapterSummary, RenderMode, RenderRequest, VoiceOut } from "../api/types";

/** Pure helpers for the Render screen: the plain names for the two recording modes, the
    single-narrator spec every estimate/quote/render must carry identically, the whole-book
    confirm the screen asks BEFORE posting, and the audio-check row's suggested word. */

/* -------------------------------------------------- modes */

export const MODE_LABEL: Record<RenderMode, string> = {
  multivoice: "Full cast",
  single: "One narrator",
};

/** The archive names the render summary uses ("multi"/"single") in the same words. */
export const ARCHIVE_LABEL: Record<"multi" | "single", string> = {
  multi: MODE_LABEL.multivoice,
  single: MODE_LABEL.single,
};

/* -------------------------------------------------- one narrator: voice + speed */

/** Narrator picker value: the server's standard narrator, a built-in Kokoro preset (by its
    engine id), or a saved library voice (by voice_id). */
export type NarratorChoice = "default" | `preset:${string}` | `voice:${string}`;

export type SingleSpec = NonNullable<RenderRequest["single"]>;

export const SPEEDS: { value: number; label: string }[] = [
  { value: 0.8, label: "slower" },
  { value: 0.9, label: "a little slower" },
  { value: 1, label: "normal" },
  { value: 1.1, label: "a little faster" },
  { value: 1.25, label: "faster" },
];

/** Does the speed control apply to this narrator? A SAVED library voice renders with its own
    stored settings (the server ignores a speed sent with it), so only the standard narrator
    and the built-in presets take a speed. */
export function narratorTakesSpeed(choice: NarratorChoice): boolean {
  return !choice.startsWith("voice:");
}

/** The `single` spec for estimate, quote and render — one function so all three always
    agree (the cost token binds the exact work). Normal speed is left out, so the standard
    narrator at normal speed is `{}`, exactly the server default. */
export function narratorSpec(
  choice: NarratorChoice,
  speed: number,
  voices: readonly Pick<VoiceOut, "voice_id" | "engine">[] | undefined,
): SingleSpec {
  const withSpeed = speed !== 1 ? { speed } : {};
  if (choice === "default") return { ...withSpeed };
  if (choice.startsWith("preset:")) return { engine: "kokoro", voice: choice.slice("preset:".length), ...withSpeed };
  const id = choice.slice("voice:".length);
  const engine = voices?.find((v) => v.voice_id === id)?.engine;
  return engine ? { engine, voice: id } : { voice: id };
}

/** Query params for GET /cost-estimate (the same fields as the POST bodies' `single`). */
export function singleParams(spec: SingleSpec | undefined): string {
  if (!spec) return "";
  const q = new URLSearchParams();
  if (spec.engine) q.set("engine", spec.engine);
  if (spec.voice) q.set("voice", spec.voice);
  if (spec.speed !== undefined) q.set("speed", String(spec.speed));
  const s = q.toString();
  return s ? `&${s}` : "";
}

/** The picker value that matches the voice an existing one-narrator recording used, so a
    chapter-range re-make keeps the same narrator (the server refuses to mix narrators). */
export function choiceForVoice(
  voiceId: string | null | undefined,
  voices: readonly Pick<VoiceOut, "voice_id">[] | undefined,
  presetIds: readonly string[] | undefined,
): NarratorChoice | null {
  if (!voiceId) return null;
  if (voices?.some((v) => v.voice_id === voiceId)) return `voice:${voiceId}`;
  if (presetIds?.includes(voiceId)) return `preset:${voiceId}`;
  return null;
}

/** The voice id a choice records with, when the screen can tell (the standard narrator
    needs the server's default preset id). */
export function choiceVoiceId(choice: NarratorChoice, defaultVoice: string | undefined): string | null {
  if (choice === "default") return defaultVoice ?? null;
  return choice.slice(choice.indexOf(":") + 1);
}

/* -------------------------------------------------- whole-book confirm */

export interface WholeBookDetail {
  speakable_blocks: number;
  runtime_estimate_seconds: number;
}

/** The whole-book stop the server enforces (routes/render.py: an empty chapter list over
    `limits.full_render_confirm_blocks` speakable paragraphs needs `confirm_full`), computed
    BEFORE the request so the one confirmation goes out with it. Null when no stop applies,
    or when the limit isn't known (the server's 409 then still asks, as a fallback). */
export function wholeBookConfirm(
  scopeChapters: readonly number[],
  chapters: readonly Pick<ChapterSummary, "speakable_blocks">[] | null | undefined,
  limit: number | undefined,
  runtimeSeconds: number | null | undefined,
): WholeBookDetail | null {
  if (scopeChapters.length > 0 || !chapters || limit === undefined) return null;
  const speakable = chapters.reduce((a, c) => a + c.speakable_blocks, 0);
  if (speakable <= limit) return null;
  return { speakable_blocks: speakable, runtime_estimate_seconds: runtimeSeconds ?? 0 };
}

/* -------------------------------------------------- audio checks */

const strip = (w: string) => w.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");

/** Failures a pronunciation entry can't fix: the scorer's number, negation, meaning, loop and
    cut-off reasons (validate/scoring.py). For these no word is suggested. */
const NOT_PRONUNCIATION = /number|negation|meaning flipped|repeated words|cut off/i;

/** Words never worth teaching: number words (a wrong or missing number is a misread, and the
    check may hear "twelve" as "12") and the short common words a dropped clause leaves. */
const SKIP = new Set(
  (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " +
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand " +
    "million billion first second third half quarter point " +
    "the and but not nor for with from that this then than they them their there were was are has had " +
    "have his her him she you your our its into onto over upon what when where which who whom why how " +
    "all any can could did does don't would should will shall just said says very been being"
  ).split(" "),
);

/** The word to suggest on the Pronunciation screen for a failed audio check: the first word
    of the text that the check did not hear ("Kael'thas raised…" heard "Kale has raised…" ->
    "Kael'thas"). Only a suggestion, and only when the failure could be one — not for a
    number, negation, loop or cut-off reason, never a number or short common word, and never
    a word that is already a lexicon respelling (the text checked is the NORMALIZED text, so a
    taught word appears in its respelled form). Null when nothing fits. */
export function misheardWord(
  expected: string,
  heard: string,
  reason?: string | null,
  respelled?: ReadonlySet<string>,
): string | null {
  if (reason && NOT_PRONUNCIATION.test(reason)) return null;
  const got = new Set(heard.split(/\s+/).map(strip).filter(Boolean));
  for (const raw of expected.split(/\s+/)) {
    const w = strip(raw);
    if (!w || got.has(w) || w.length < 3 || /\p{N}/u.test(w) || SKIP.has(w) || respelled?.has(w)) continue;
    return raw.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
  }
  return null;
}

/** The lowercased words of every lexicon respelling, for misheardWord's `respelled`. */
export function respelledWords(entries: readonly { respelling: string }[] | undefined): Set<string> {
  return new Set((entries ?? []).flatMap((e) => e.respelling.split(/\s+/).map(strip).filter(Boolean)));
}
