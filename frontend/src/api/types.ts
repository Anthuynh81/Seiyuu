/* TypeScript mirrors of the M6b API schemas (src/seiyuu/api/schemas.py). Only what the
   built screens consume — grown per section, never speculatively. */

export interface ErrorEnvelope {
  error: { code: string; message: string; detail: unknown };
}

export type JobState = "queued" | "running" | "succeeded" | "failed" | "canceled";
export type JobKind = "ingest" | "attribute" | "render" | "assemble" | "master" | "warmup";

export interface JobOut {
  job_id: string;
  book_id: string;
  kind: JobKind;
  state: JobState;
  progress_text: string;
  error: string | null;
  cancel_requested: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  is_terminal: boolean;
  params: Record<string, unknown> | null;
}

export interface JobsOut {
  jobs: JobOut[];
}

export interface ActiveJobSummary {
  job_id: string;
  kind: JobKind;
  state: "queued" | "running";
}

export interface BookCard {
  book_id: string;
  title: string | null;
  authors: string[];
  ingested: boolean;
  attributed: boolean;
  assigned: boolean;
  rendered: boolean;
  assembled: boolean;
  mastered: boolean;
  active_job: ActiveJobSummary | null;
}

export interface BooksOut {
  books: BookCard[];
}

// -- per-book pronunciation lexicon ----------------------------------------------------------

/** One pronunciation override. `respelling` is spoken on every engine; `ipa` (optional) is
    honored ONLY on the Kokoro profile and ignored on validated engines. */
export interface LexiconEntry {
  term: string;
  respelling: string;
  ipa: string | null;
  note: string | null;
  case_sensitive: boolean;
}

export interface SuggestedTerm {
  term: string;
  count: number;
  sample: string;
}

export interface LexiconOut {
  book_id: string;
  schema_version: number;
  entries: LexiconEntry[];
  suggestions: SuggestedTerm[];
}

export interface LexiconSaved {
  book_id: string;
  schema_version: number;
  entries: LexiconEntry[];
  affected_blocks: number;
  total_speakable_blocks: number;
}

export interface LexiconPreviewOut {
  affected_blocks: number;
  total_speakable_blocks: number;
}

/** F3 (v1.1): one ADVISORY LLM-proposed grapheme respelling. The user accepts it into the
    lexicon (which stays the deterministic source of truth); the LLM never writes it. */
export interface RespellSuggestion {
  term: string;
  respelling: string;
  note: string | null;
}

export interface RespellSuggestOut {
  provider: string;
  model: string;
  suggestions: RespellSuggestion[];
}

// -- delete a book (F3) ----------------------------------------------------------------------

/** DELETE /api/books/{id} success — what was actually torn down. */
export interface BookDeletedOut {
  book_id: string;
  output_removed: boolean;
  books_removed: boolean;
  jobs_rows_deleted: number;
  paid_segments_discarded: number;
}

/** 402 payment_confirmation_required detail: the paid cloud renders a delete would
    discard. Re-send the DELETE with confirm_paid=true to proceed. */
export interface PaidArtifacts {
  paid_segment_count: number;
  engines: string[];
  paid_voice_ids: string[];
  estimated_usd: number | null;
}

export interface IngestResponse {
  book: Omit<BookCard, "active_job">;
  chapters: number;
  blocks: number;
  skipped_items: string[];
  dropped_sections: string[];
}

export interface ChapterSummary {
  index: number;
  title: string;
  blocks: number;
  speakable_blocks: number;
}

export interface FileDownload {
  url: string;
  bytes: number;
}

export interface ChapterDownload extends FileDownload {
  index: number;
}

/** BookDetail.cover and the PUT /cover response (mirrors CoverOut in schemas.py). */
export interface CoverOut {
  content_type: string;
  bytes: number;
}

export interface BookDetail {
  status: Omit<BookCard, "active_job">;
  chapters: ChapterSummary[] | null;
  runtime_estimate_seconds: number | null;
  active_job: ActiveJobSummary | null;
  recent_jobs: JobOut[];
  downloads: { m4b: FileDownload | null; chapter_mp3s: ChapterDownload[] };
  cover: CoverOut | null;
}

export type RenderMode = "multivoice" | "single";

export interface CostEstimateOut {
  total_usd: number;
  paid_segments: number;
  cached_segments: number;
  free_segments: number;
  fingerprint: string;
  assignment_hash: string | null;
  mode: RenderMode;
  chapters: number[];
  edit_warnings: string[];
}

export interface QuoteResponse {
  token: string;
  book_id: string;
  chapters: number[];
  total_usd: number;
  paid_segments: number;
  fingerprint: string;
  assignment_hash: string | null;
  issued_at: number;
  expires_at: number;
  ttl_seconds: number;
  max_usd_ceiling: number;
}

export interface RenderRequest {
  mode: RenderMode;
  chapters?: number[];
  cost_token?: string;
  confirm_full?: boolean;
  single?: { engine?: string; voice?: string; speed?: number; seed?: number };
  // F2b: per-render emotion override (undefined -> the server default cfg.apply_emotion)
  apply_emotion?: boolean;
}

export interface RenderSummaryOut {
  book_id: string;
  mode: RenderMode;
  chapters: { index: number; title: string; segments: number; duration_seconds: number }[];
  total_seconds: number;
  voices_used: Record<string, { engine: string; engine_model_version: string; kind: string }>;
  validation_failures: number;
}

export interface ValidationRow {
  chapter_index: number;
  block_id: string;
  segment_index: number;
  voice_id: string | null;
  ok: boolean;
  score: number;
  expected: string;
  transcript: string;
  synth_attempts: number;
}

export interface ValidationReportOut {
  validated_segments: number;
  validation_failures: number;
  results: ValidationRow[];
}

// -- character review ----------------------------------------------------------------------

export interface CharacterSummary {
  id: string;
  name: string;
  aliases: string[];
  gender: string | null;
  age_hint: string | null;
  line_count: number;
  sample_lines: string[];
  first_appearance: string | null; // block id like "ch013_b0042"
}

export interface FlaggedBlock {
  block_id: string;
  chapter_index: number;
  reason: string;
}

export interface CharactersOverview {
  book_id: string;
  provider_id: string;
  model_id: string;
  prompt_version: string;
  narration_segments: number;
  low_confidence_segments: number;
  /** quoted spans nobody attributed — narrator-voiced, surfaced apart from narration */
  unattributed_quote_segments: number;
  confidence_threshold: number;
  characters: CharacterSummary[];
  flagged: FlaggedBlock[];
  notes: string[];
  edit_warnings: string[];
}

export interface SegmentRow {
  block_id: string;
  segment_index: number;
  type: "narration" | "dialogue" | "thought";
  speaker: string | null;
  speaker_name: string | null;
  text: string;
  confidence: number;
  /** speaker is null but the text is a quoted span — an unattributed quote, not narration */
  unattributed_quote: boolean;
  has_audio: boolean;
  audio_segment: number | null; // ?segment= index for the audio route; null = no timing
  duration_seconds: number | null;
  voice_id: string | null; // the voice that actually rendered this row's audio
  audio_key: string | null; // wav SegmentKey hash — render identity + cache-buster
}

export interface SegmentBrowserOut {
  chapter_index: number;
  title: string;
  segments: SegmentRow[];
  edit_warnings: string[];
}

// -- per-segment emotion (Phase 1 F2) --------------------------------------------------------

/** The closed, quantized emotion taxonomy the v5/v6 attribution prompt emits (mirrors
    `EmotionLabel` in attribute/models.py). NEUTRAL degrades to no render override. */
export type EmotionLabel = "neutral" | "happy" | "sad" | "angry" | "fearful" | "tender" | "tense";

/** One dialogue segment's emotion tag. `intensity` is a 3-level scale (1=low … 3=high). */
export interface EmotionVerdict {
  label: EmotionLabel;
  intensity: number;
}

/** Minimal view of GET /api/books/{id}/attribution — only what the Review screen reads to
    surface per-segment emotion. `segment_emotions` is index-aligned to `segments` (both are
    regenerated together in the same attribution.json, so they can never desync); it is empty
    on a pre-emotion (v3/v4) report, which the UI treats as "no tags". */
export interface AttributionSegmentView {
  block_id: string;
}

export interface AttributedChapterView {
  index: number; // 1-based
  segments: AttributionSegmentView[];
  segment_emotions: (EmotionVerdict | null)[];
}

export interface AttributionReportView {
  chapters: AttributedChapterView[];
}

export interface AttributionOut {
  report: AttributionReportView;
  edit_warnings: string[];
}

// -- word-exact read-along (F2) --------------------------------------------------------------

/** One whisper-timed spoken token, seconds within the clip's wav. */
export interface WordTiming {
  start: number;
  end: number;
  word: string;
}

/** GET /api/books/{id}/segments/{block_id}/words?segment=N — whisper word timings for one
    rendered wav. 404 (scene break / not-yet-rendered / missing wav) degrades to the
    length-interpolated fallback. */
export interface SegmentWords {
  words: WordTiming[];
  audio_duration: number;
  source: string;
}

export type EditRequest =
  | { op: "rename"; character_id: string; new_name: string }
  | { op: "merge"; loser_id: string; winner_id: string }
  | { op: "reassign"; block_id: string; segment_index: number; speaker: string | null }
  // F2a: set (verdict) or clear (null) one segment's emotion overlay
  | { op: "set_emotion"; block_id: string; segment_index: number; emotion: EmotionVerdict | null };

export interface EditLog {
  version: number;
  ops: Record<string, unknown>[];
}

// -- casting (voice assignment) --------------------------------------------------------------

export interface VoiceAssignment {
  schema_version: number;
  book_id: string;
  stage: "draft" | "final";
  narrator_voice_id: string;
  assignments: Record<string, string>; // character_id -> voice_id
  thought_voice_id: string | null; // null = thoughts use the speaker's own voice
  created_at: string;
}

export interface AssignmentDraftResponse {
  assignment: VoiceAssignment;
  created_voice_ids: string[];
  edit_warnings: string[];
}

export type CastStrategy = "hash" | "smart";

export interface SuggestCastResponse {
  assignment: VoiceAssignment;
  would_create_voice_ids: string[];
  would_recast_voice_ids: string[];
  edit_warnings: string[];
}

export interface AssignmentWrite {
  stage: "draft" | "final";
  narrator_voice_id: string;
  assignments: Record<string, string>;
  thought_voice_id: string | null;
}

// -- series / library voice consistency (F5) -------------------------------------------------

/** One declared series. `voice_links` maps a cross-book identity key (casefolded canonical
    name) -> library voice_id; `book_ids` is a plain membership list. Cross-book matching is
    scoped to THIS series only — there is no global name match. */
export interface Series {
  series_id: string;
  name: string;
  book_ids: string[];
  voice_links: Record<string, string>; // identity_key -> voice_id
}

export interface SeriesListOut {
  series: Series[];
}

/** A within-series name match surfaced for the user to CONFIRM (never auto-applied). Character
    `character_id` in the joining book matches an existing link. `voice_exists` is false when the
    linked voice was deleted — the UI shows it unavailable and it can't be inherited. */
export interface LinkSuggestion {
  character_id: string;
  canonical_name: string;
  identity_key: string;
  voice_id: string;
  voice_exists: boolean;
}

export interface LinkSuggestionsOut {
  series_id: string;
  book_id: string;
  suggestions: LinkSuggestion[];
}

/** Result of an explicit save-to-series write-back: the updated series plus the identity keys
    that were added or updated by folding the book's cast in. */
export interface SaveCastOut {
  series: Series;
  linked_keys: string[];
}

// -- voice studio --------------------------------------------------------------------------

export interface VoiceOut {
  voice_id: string;
  name: string;
  kind: "preset" | "blend" | "cloned";
  engine: string;
  preset_id: string | null;
  blend: { preset_id: string; weight: number }[] | null;
  reference_audio: string | null;
  seed: number;
  consent_attested: boolean;
  consent: { attested_by: string; reference_sha256: string; attested_at: string } | null;
  tags: string[]; // free-form; auto-cast stamps ["auto", book_id]
  created_at: string;
  has_audition: boolean;
}

export interface VoiceListOut {
  voices: VoiceOut[];
  unreadable: { voice_id: string; error: string }[];
}

export type VoiceCreate =
  | { kind: "preset"; name: string; engine: string; preset_id: string; voice_id?: string }
  | {
      kind: "blend";
      name: string;
      /** Manual mix: ≥2 layers, weights are ratios (server-normalized). Omit for the auto recipe. */
      components?: { preset_id: string; weight: number }[];
      gender?: string | null;
      accent?: "a" | "b";
      voice_id?: string;
    };

export interface AuditionOut {
  voice_id: string;
  duration_seconds: number;
  cost_usd: number;
  audition_url: string;
}

export interface VoiceReferencesOut {
  voice_id: string;
  references: { book_id: string; role: string }[];
}

export interface CloudSlotsOut {
  max_slots: number;
  count: number;
  slots: { voice_id: string; cloud_id: string; seq: number }[];
}

export interface EngineVoicesOut {
  engine_id: string;
  voices: {
    id: string;
    name: string;
    language: string | null;
    gender: string | null;
    description: string | null;
  }[];
}

export interface AttributionDefaults {
  provider: string;
  model: string;
  anthropic_model: string;
  prompt_version: string;
  hybrid: boolean;
}

/** Slim view of GET /api/system — only what screens consume. */
export interface SystemStatusOut {
  attribution: AttributionDefaults;
  keys: { anthropic_configured: boolean; elevenlabs_configured: boolean };
  apply_emotion: boolean; // F2b: server default for the per-render emotion toggle
}

/** "ch013_b0042" -> 13; null when unparsable. */
export function chapterOfBlock(blockId: string | null): number | null {
  const m = blockId?.match(/^ch(\d+)_/);
  return m ? Number(m[1]) : null;
}

/** The six pipeline stages in signal-path order, with the card flag for each. */
export const STAGES = [
  ["ingested", "ingest"],
  ["attributed", "attribute"],
  ["assigned", "assign"],
  ["rendered", "render"],
  ["assembled", "assemble"],
  ["mastered", "master"],
] as const;

/** Which stage a running job kind lights up on the signal path. */
export const KIND_STAGE: Partial<Record<JobKind, (typeof STAGES)[number][0]>> = {
  attribute: "attributed",
  render: "rendered",
  assemble: "assembled",
  master: "mastered",
};
