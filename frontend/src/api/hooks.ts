import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";

import { api, ApiError, postForm, postJson } from "./client";
import type {
  ArchivedRenderMode,
  AssignmentDraftResponse,
  AssignmentWrite,
  AttributionOut,
  AuditionOut,
  CastStrategy,
  BookDeletedOut,
  BookDetail,
  BooksOut,
  ChapterWordsOut,
  CharactersOverview,
  CloudSlotsOut,
  CostEstimateOut,
  CoverOut,
  EditLog,
  EditRequest,
  EngineVoicesOut,
  IngestResponse,
  JobOut,
  JobsOut,
  LexiconEntry,
  LexiconOut,
  LexiconPreviewOut,
  LexiconSaved,
  QuoteResponse,
  LinkSuggestionsOut,
  RenderMode,
  RespellSuggestOut,
  RenderRequest,
  RenderSummaryOut,
  SaveCastOut,
  SegmentBrowserOut,
  SegmentWords,
  Series,
  SeriesListOut,
  SuggestCastResponse,
  SystemStatusOut,
  ValidationReportOut,
  VoiceAssignment,
  VoiceCreate,
  VoiceListOut,
  VoiceOut,
  VoiceReferencesOut,
} from "./types";

/** Polling discipline: this is THE live poll — book payloads deliberately carry no
    progress, so everything watching a job watches this one query. */
export function useLiveJobs() {
  return useQuery({
    queryKey: ["jobs", "live"],
    queryFn: () => api<JobsOut>("/api/jobs?state=queued&state=running"),
    refetchInterval: 2000,
    // keep polling in hidden tabs: JobCompletionWatcher and the finish chain both clock
    // off this poll, and "keep this page open" must not secretly mean "and visible"
    refetchIntervalInBackground: true,
  });
}

export function useBooks() {
  // Stable key: JobCompletionWatcher invalidates on job transitions, and a same-key
  // refetch keeps old data on screen — no undefined window, no derived-bookId flicker.
  return useQuery({
    queryKey: ["books"],
    queryFn: () => api<BooksOut>("/api/books"),
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api<JobOut>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useIngest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return postForm<IngestResponse>("/api/books", form);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["books"] }),
  });
}

/** Delete a whole book. `confirm_paid` gates the discard of paid cloud renders: the first
    call sends false; a 402 (payment_confirmation_required, detail = PaidArtifacts) lets the
    UI escalate by re-sending true. 409 (conflicting_job) and 500 (partial_delete) propagate
    to the caller as ApiError. */
export function useDeleteBook() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ bookId, confirmPaid }: { bookId: string; confirmPaid: boolean }) =>
      api<BookDeletedOut>(`/api/books/${encodeURIComponent(bookId)}?confirm_paid=${confirmPaid}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["books"] });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// -- render & jobs ------------------------------------------------------------------------

export function useBook(bookId: string | null) {
  // Stable key (see useBooks): invalidation-driven freshness instead of key churn.
  return useQuery({
    queryKey: ["book", bookId],
    queryFn: ({ signal }) => api<BookDetail>(`/api/books/${bookId}`, { signal }),
    enabled: bookId !== null,
  });
}

export function useBookJobs(bookId: string | null) {
  return useQuery({
    queryKey: ["jobs", "book", bookId],
    queryFn: () => api<JobsOut>(`/api/jobs?book_id=${encodeURIComponent(bookId!)}&limit=25`),
    enabled: bookId !== null,
    refetchInterval: 2500,
    refetchIntervalInBackground: true, // the assemble->master finish chain advances on this poll
  });
}

function chapterParams(chapters: number[]): string {
  return chapters.map((c) => `&chapters=${c}`).join("");
}

export function useEstimate(
  bookId: string | null,
  mode: RenderMode,
  chapters: number[],
  ready: boolean,
  applyEmotion?: boolean, // F2b: undefined -> server default; must match what's minted/rendered
  force = false, // re-render: price cached in-scope segments as billable work (parity with render)
) {
  const emo = applyEmotion === undefined ? "" : `&apply_emotion=${applyEmotion}`;
  const frc = force ? "&force=true" : "";
  return useQuery({
    queryKey: ["estimate", bookId, mode, chapters, applyEmotion ?? null, force],
    queryFn: () =>
      api<CostEstimateOut>(
        `/api/books/${bookId}/cost-estimate?mode=${mode}${chapterParams(chapters)}${emo}${frc}`,
      ),
    enabled: bookId !== null && ready,
  });
}

/** Which chapters already have rendered audio (404 until the first render = none). */
export function useRenderSummary(bookId: string | null, rendered: boolean) {
  return useQuery({
    queryKey: ["render-summary", bookId, rendered],
    queryFn: () => api<RenderSummaryOut>(`/api/books/${bookId}/render`),
    enabled: bookId !== null && rendered,
  });
}

export function useValidation(bookId: string | null, rendered: boolean) {
  return useQuery({
    queryKey: ["validation", bookId],
    queryFn: () => api<ValidationReportOut>(`/api/books/${bookId}/validation`),
    enabled: bookId !== null && rendered,
  });
}

/** Point manifest.json — the render truth Listen/assemble/master read — at the chosen mode's
    archived render: an atomic pointer move on the server, no synthesis, no cache touch.
    Invalidates every reader of the active manifest: the render summary (Render & Jobs AND
    Listen's provenance/chapter list), the segment rows (has_audio/audio_key drive the
    read-along; word timings are content-addressed by audio_key so they follow the refetch),
    the validation report, and the book detail. A 409 conflicting_job (a render/assemble/
    master job owns the manifest right now) surfaces as ApiError for the control to render. */
export function useSwitchRenderMode(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (mode: ArchivedRenderMode) =>
      postJson<RenderSummaryOut>(`/api/books/${bookId}/render/mode`, { mode }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["render-summary", bookId] });
      qc.invalidateQueries({ queryKey: ["segments", bookId] });
      qc.invalidateQueries({ queryKey: ["validation", bookId] });
      qc.invalidateQueries({ queryKey: ["book", bookId] });
    },
  });
}

export function useMintQuote(bookId: string) {
  return useMutation({
    mutationFn: ({
      mode,
      chapters,
      applyEmotion,
      force,
    }: {
      mode: RenderMode;
      chapters: number[];
      applyEmotion?: boolean; // F2b: bound into the quote fingerprint; must match the render
      force?: boolean; // re-render: quote must be minted force=True to authorize a forced render
    }) =>
      postJson<QuoteResponse>(`/api/books/${bookId}/quotes`, {
        mode,
        chapters,
        ...(mode === "single" ? { single: {} } : {}),
        ...(applyEmotion === undefined ? {} : { apply_emotion: applyEmotion }),
        ...(force ? { force: true } : {}),
      }),
  });
}

export interface AttributeInput {
  chapters: number[]; // [] = whole book; a subset merges into the existing report
  provider?: "local" | "anthropic";
  model?: string;
  confirm_paid?: boolean; // anthropic runs are paid — the enqueue 402s without this
}

export function useAttribute(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: AttributeInput) => postJson<JobOut>(`/api/books/${bookId}/attribute`, input),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["book", bookId] });
    },
  });
}

// -- pronunciation lexicon -----------------------------------------------------------------

export function useLexicon(bookId: string | null) {
  return useQuery({
    queryKey: ["lexicon", bookId],
    queryFn: () => api<LexiconOut>(`/api/books/${bookId}/lexicon`),
    enabled: bookId !== null,
  });
}

/** Save the whole lexicon. Returns the affected-segment count vs the previous save so the UI
    can tell the user how many segments the change re-synthesizes. Editing invalidates the
    estimate (normalized_text_hash shifts) and the book (a lexicon is per-book input). */
export function useSaveLexicon(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (entries: LexiconEntry[]) =>
      api<LexiconSaved>(`/api/books/${bookId}/lexicon`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entries }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lexicon", bookId] });
      qc.invalidateQueries({ queryKey: ["estimate", bookId] });
    },
  });
}

/** Affected-segment count for a PROPOSED lexicon, without saving — shown before commit. */
export function usePreviewLexicon(bookId: string) {
  return useMutation({
    mutationFn: (entries: LexiconEntry[]) =>
      postJson<LexiconPreviewOut>(`/api/books/${bookId}/lexicon/preview`, { entries }),
  });
}

export interface RespellInput {
  terms?: string[]; // omit/empty -> the backend uses the deterministic hard-name suggestions
  provider?: string; // override the configured provider ("local" free | "anthropic" PAID)
  confirm_paid?: boolean; // required when the resolved provider is anthropic (paid)
}

/** F3 (v1.1): opt-in LLM respelling suggestions for hard terms. ADVISORY — writes nothing; the
    user folds accepted respellings into the editor (see lib/applyRespellings) and saves. A 402
    means the resolved provider is anthropic and confirm_paid is required. */
export function useSuggestRespellings(bookId: string) {
  return useMutation({
    mutationFn: (input: RespellInput = {}) =>
      postJson<RespellSuggestOut>(`/api/books/${bookId}/lexicon/suggest-respellings`, input),
  });
}

export function useSystem() {
  return useQuery({
    queryKey: ["system"],
    queryFn: () => api<SystemStatusOut>("/api/system"),
    staleTime: 60_000, // config facts; no need to poll
  });
}

// -- character review ----------------------------------------------------------------------

export function useCharacters(bookId: string | null, attributed: boolean) {
  return useQuery({
    queryKey: ["characters", bookId],
    queryFn: () => api<CharactersOverview>(`/api/books/${bookId}/characters?sample_lines=1`),
    enabled: bookId !== null && attributed,
  });
}

export function useSegments(bookId: string | null, chapter: number, attributed: boolean) {
  return useQuery({
    queryKey: ["segments", bookId, chapter],
    // signal: rapid chapter paging aborts the superseded fetch instead of letting the
    // server finish a whole-chapter payload nobody will read
    queryFn: ({ signal }) =>
      api<SegmentBrowserOut>(`/api/books/${bookId}/chapters/${chapter}/segments`, { signal }),
    enabled: bookId !== null && attributed,
  });
}

/** The chapter's effective attribution report, trimmed to one chapter (`?chapters=`), for the
    per-segment emotion side-channel the Review screen shows (Phase 1 F2). Read-only: emotion is
    captured by v5/v6 attribution regardless of the server's opt-in `apply_emotion` render flag.
    Keyed under "attribution" so a recorded edit (invalidateReview) also refreshes it. */
export function useChapterAttribution(bookId: string | null, chapter: number, attributed: boolean) {
  return useQuery({
    queryKey: ["attribution", bookId, chapter],
    queryFn: ({ signal }) =>
      api<AttributionOut>(`/api/books/${bookId}/attribution?chapters=${chapter}`, { signal }),
    enabled: bookId !== null && attributed,
  });
}

/** One clip = one rendered wav we want whisper word-timings for. `audioKey` is the wav's
    SegmentKey hash: it changes iff the audio content changes, so it doubles as the
    react-query cache key AND the cache-buster (a re-render re-fetches, never serves stale). */
export interface SegmentWordsClip {
  key: string; // `${block_id}:${audio_segment}` — matches the player clip key
  blockId: string;
  segment: number;
  audioKey: string | null;
}

export interface SegmentWordsResult {
  /** Only clips whose words resolved (200) land here; a 404 or still-loading clip is absent,
      and the read-along keeps its length-interpolated fallback for it. */
  byKey: Map<string, SegmentWords>;
  /** Stable signature of which clips have resolved — cheap dependency for the apply effect. */
  sig: string;
}

/** Signature of which clips have resolved AND at what audio identity. Folding `audioKey` in
    means a re-render / reassignment that swaps a clip's wav (same clip key, new audio_key)
    flips the signature even when the SET of resolved keys is unchanged — so the read-along
    re-applies the fresh whisper timings instead of silently keeping the old ones. Pure and
    order-independent (sorted). */
export function resolvedSig(resolved: { key: string; audioKey: string | null }[]): string {
  return resolved
    .map((r) => `${r.key}@${r.audioKey ?? ""}`)
    .sort()
    .join("|");
}

/** Fetch whisper word-timings for every clip in a chapter in ONE batch request (one
    manifest parse server-side, instead of an HTTP request — and a manifest parse — per
    wav). Clips the server omitted (missing wav / failed alignment) or a 404 for the whole
    chapter (not yet rendered) simply stay on interpolation. The key folds every clip's
    audio_key, so a re-render (new audio identity) refetches; content-addressed, so
    staleTime is infinite. */
export function useSegmentWords(
  bookId: string | null,
  chapter: number,
  clips: SegmentWordsClip[],
): SegmentWordsResult {
  const clipsSig = resolvedSig(clips);
  const q = useQuery({
    queryKey: ["segment-words", bookId, chapter, clipsSig],
    queryFn: async ({ signal }): Promise<ChapterWordsOut | null> => {
      try {
        return await api<ChapterWordsOut>(`/api/books/${bookId}/chapters/${chapter}/words`, {
          signal,
        });
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null; // not rendered — interpolate
        throw e;
      }
    },
    enabled: bookId !== null && clips.length > 0,
    staleTime: Infinity,
  });
  return useMemo(() => {
    const byKey = new Map<string, SegmentWords>();
    const resolved: { key: string; audioKey: string | null }[] = [];
    for (const c of clips) {
      const w = q.data?.words[c.key];
      if (w) {
        byKey.set(c.key, w);
        resolved.push({ key: c.key, audioKey: c.audioKey });
      }
    }
    return { byKey, sig: resolvedSig(resolved) };
  }, [q.data, clips]);
}

export function useEditLog(bookId: string | null) {
  return useQuery({
    queryKey: ["edits", bookId],
    queryFn: () => api<EditLog>(`/api/books/${bookId}/edits`),
    enabled: bookId !== null,
  });
}

function invalidateReview(qc: ReturnType<typeof useQueryClient>, bookId: string) {
  for (const key of ["characters", "segments", "attribution", "edits", "estimate"] as const) {
    qc.invalidateQueries({ queryKey: [key, bookId] });
  }
}

export function useRecordEdit(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (edit: EditRequest) => postJson<Record<string, unknown>>(`/api/books/${bookId}/edits`, edit),
    onSuccess: () => invalidateReview(qc, bookId),
  });
}

export function useUndoEdit(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api<{ removed: Record<string, unknown> }>(`/api/books/${bookId}/edits/last`, { method: "DELETE" }),
    onSuccess: () => invalidateReview(qc, bookId),
  });
}

export function useStartJob(bookId: string, path: "render" | "assemble" | "master") {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RenderRequest | Record<string, never>) =>
      postJson<JobOut>(`/api/books/${bookId}/${path}`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

/** Cover art for the .m4b master (and the Listen shelf). PUT multipart replaces; DELETE is
    idempotent. Both invalidate the book detail so `cover` refreshes everywhere — but the
    GET /cover URL itself never changes, so the caller must cache-bust its preview (the
    audio_key ?v= trick). 415/413/409 surface as ApiError for the control to render. */
export function useUploadCover(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return api<CoverOut>(`/api/books/${bookId}/cover`, { method: "PUT", body: form });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["book", bookId] }),
  });
}

export function useDeleteCover(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api<void>(`/api/books/${bookId}/cover`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["book", bookId] }),
  });
}

// -- casting (voice assignment) --------------------------------------------------------------

export function useAssignment(bookId: string | null, attributed: boolean) {
  return useQuery({
    queryKey: ["assignment", bookId],
    queryFn: async () => {
      try {
        return await api<VoiceAssignment>(`/api/books/${bookId}/assignment`);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null; // not cast yet
        throw e;
      }
    },
    enabled: bookId !== null && attributed,
  });
}

function invalidateCasting(qc: ReturnType<typeof useQueryClient>, bookId: string) {
  qc.invalidateQueries({ queryKey: ["assignment", bookId] });
  qc.invalidateQueries({ queryKey: ["voices"] }); // draft creates auto-blend voices
  qc.invalidateQueries({ queryKey: ["books"] }); // the assigned stage flag flips
  qc.invalidateQueries({ queryKey: ["book", bookId] });
  qc.invalidateQueries({ queryKey: ["estimate", bookId] }); // assignment hash drifts quotes
  // A cast change makes any existing render stale: refresh the summary so its
  // rendered_assignment_hash is re-read and the "re-render to hear it" banner surfaces.
  qc.invalidateQueries({ queryKey: ["render-summary", bookId] });
}

export interface DraftInput {
  strategy?: CastStrategy; // "hash" (legacy) | "smart" (collision-free book cast)
  recast?: boolean; // smart only: overwrite existing auto voices (re-renders them)
  // F5: character_id -> voice_id inherited from a series; wins over the auto-cast pick. The
  // smart caster reserves these so a new character is never cast onto a series-pinned voice.
  overrides?: Record<string, string>;
  // F4 (v1.1): smart only. Run the opt-in Layer-2 LLM caster for per-character voice-trait
  // preferences that bias the tie-breaker. cast_book still enforces distinctness/determinism, so
  // this can only change WHICH distinct voice a character takes. A 402 means the resolved cast
  // provider is anthropic and confirm_paid is required.
  use_llm?: boolean;
  cast_provider?: string; // override the configured cast provider ("local" | "anthropic")
  confirm_paid?: boolean; // required when the resolved cast provider is anthropic (paid)
}

export function useDraftAssignment(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: DraftInput = {}) =>
      postJson<AssignmentDraftResponse>(`/api/books/${bookId}/assignment/draft`, input),
    onSuccess: () => invalidateCasting(qc, bookId),
  });
}

/** PREVIEW the smart cast without writing anything — returns the proposed distinct-voice
    assignment plus which auto voices it would create vs (with recast) overwrite. Apply it by
    calling useDraftAssignment with strategy:"smart". */
export function useSuggestCast(bookId: string) {
  return useMutation({
    mutationFn: () =>
      postJson<SuggestCastResponse>(`/api/books/${bookId}/assignment/suggest`, { strategy: "smart" }),
  });
}

export function useSaveAssignment(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: AssignmentWrite) =>
      api<VoiceAssignment>(`/api/books/${bookId}/assignment`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateCasting(qc, bookId),
  });
}

// -- series / library voice consistency (F5) -------------------------------------------------

/** All declared series (global registry). */
export function useSeriesList() {
  return useQuery({ queryKey: ["series"], queryFn: () => api<SeriesListOut>("/api/series") });
}

/** Create a series SEEDED from a book — its cast becomes the initial voice_links. The book must
    be attributed and assigned (the backend 409s / 4xx otherwise, surfaced to the caller). */
export function useCreateSeries() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, bookId }: { name: string; bookId: string }) =>
      postJson<Series>("/api/series", { name, book_id: bookId }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["series"] }),
  });
}

/** Attach a sibling book to a series (idempotent). Does NOT learn its cast — that is the
    explicit save-cast action, so precision is preserved. */
export function useAddBookToSeries(seriesId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (bookId: string) =>
      postJson<Series>(`/api/series/${seriesId}/books`, { book_id: bookId }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["series"] }),
  });
}

/** Cross-book link SUGGESTIONS for a book joining a series: each character whose name matches an
    existing link, for the user to confirm. Never auto-applied. Scoped to this series only. */
export function useLinkSuggestions(seriesId: string | null, bookId: string | null, enabled: boolean) {
  return useQuery({
    queryKey: ["link-suggestions", seriesId, bookId],
    queryFn: () =>
      api<LinkSuggestionsOut>(`/api/series/${seriesId}/books/${bookId}/link-suggestions`),
    enabled: enabled && seriesId !== null && bookId !== null,
  });
}

/** Explicit write-back: fold a book's cast into the series' voice_links (last-write-wins). The
    only path that grows a series' links from a book — nothing is learned silently. */
export function useSaveCastToSeries(seriesId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (bookId: string) =>
      postJson<SaveCastOut>(`/api/series/${seriesId}/save-cast`, { book_id: bookId }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["series"] }),
  });
}

/** Remove a character's voice link by identity key (case-insensitive; idempotent). */
export function useUnlinkSeries(seriesId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      api<Series>(`/api/series/${seriesId}/links?name=${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["series"] }),
  });
}

// -- voice studio --------------------------------------------------------------------------

export function useVoices() {
  return useQuery({ queryKey: ["voices"], queryFn: () => api<VoiceListOut>("/api/voices") });
}

export function useCloudSlots() {
  return useQuery({ queryKey: ["cloud-slots"], queryFn: () => api<CloudSlotsOut>("/api/cloud-slots") });
}

export function useKokoroPresets() {
  return useQuery({
    queryKey: ["engine-voices", "kokoro"],
    queryFn: () => api<EngineVoicesOut>("/api/engines/kokoro/voices"),
    staleTime: Infinity, // hardcoded preset catalog
  });
}

export function useCreateVoice() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: VoiceCreate) => postJson<VoiceOut>("/api/voices", body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["voices"] }),
  });
}

export function useCloneVoice() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: { file: File; name: string; engine: string; attestedBy: string; replace?: boolean }) => {
      const form = new FormData();
      form.append("file", input.file);
      form.append("name", input.name);
      form.append("engine", input.engine);
      form.append("consent", "true");
      form.append("attested_by", input.attestedBy);
      if (input.replace) form.append("replace", "true");
      return postForm<VoiceOut>("/api/voices/clone", form);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["voices"] }),
  });
}

export function useAudition(voiceId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (confirmPaid: boolean) =>
      postJson<AuditionOut>(`/api/voices/${voiceId}/audition`, { confirm_paid: confirmPaid }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["voices"] });
      qc.invalidateQueries({ queryKey: ["cloud-slots"] });
    },
  });
}

export function useSetVoiceTags() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ voiceId, tags }: { voiceId: string; tags: string[] }) =>
      api<VoiceOut>(`/api/voices/${voiceId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tags }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["voices"] }),
  });
}

/** Rename a voice — a pure label change (PATCH name only). Characters reference voice_id and
    name is in no cache key, so no rendered audio drifts; recipe/seed stay immutable. */
export function useRenameVoice() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ voiceId, name }: { voiceId: string; name: string }) =>
      api<VoiceOut>(`/api/voices/${voiceId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["voices"] }),
  });
}

export function useDeleteVoice() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (voiceId: string) => {
      const refs = await api<VoiceReferencesOut>(`/api/voices/${voiceId}/references`);
      if (refs.references.length > 0) {
        const roles = refs.references.map((r) => `${r.book_id} (${r.role})`).join(", ");
        throw new ApiError(409, "voice_referenced", `still assigned in: ${roles}`);
      }
      return api<{ deleted: string }>(`/api/voices/${voiceId}`, { method: "DELETE" });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["voices"] }),
  });
}

export function useWarmup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (engineId: string) => api<JobOut>(`/api/engines/${engineId}/warmup`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}
