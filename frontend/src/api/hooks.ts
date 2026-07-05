import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, ApiError, postForm, postJson } from "./client";
import type {
  AssignmentDraftResponse,
  AssignmentWrite,
  AuditionOut,
  BookDetail,
  BooksOut,
  CharactersOverview,
  CloudSlotsOut,
  CostEstimateOut,
  EditLog,
  EditRequest,
  EngineVoicesOut,
  IngestResponse,
  JobOut,
  JobsOut,
  QuoteResponse,
  RenderMode,
  RenderRequest,
  RenderSummaryOut,
  SegmentBrowserOut,
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
  });
}

export function useBooks() {
  const live = useLiveJobs();
  const liveCount = live.data?.jobs.length ?? 0;
  return useQuery({
    queryKey: ["books", liveCount], // a job finishing (count drops) refetches the shelf
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

// -- render & jobs ------------------------------------------------------------------------

export function useBook(bookId: string | null) {
  const live = useLiveJobs();
  const liveCount = live.data?.jobs.length ?? 0;
  return useQuery({
    queryKey: ["book", bookId, liveCount],
    queryFn: () => api<BookDetail>(`/api/books/${bookId}`),
    enabled: bookId !== null,
  });
}

export function useBookJobs(bookId: string | null) {
  return useQuery({
    queryKey: ["jobs", "book", bookId],
    queryFn: () => api<JobsOut>(`/api/jobs?book_id=${encodeURIComponent(bookId!)}&limit=25`),
    enabled: bookId !== null,
    refetchInterval: 2500,
  });
}

function chapterParams(chapters: number[]): string {
  return chapters.map((c) => `&chapters=${c}`).join("");
}

export function useEstimate(bookId: string | null, mode: RenderMode, chapters: number[], ready: boolean) {
  return useQuery({
    queryKey: ["estimate", bookId, mode, chapters],
    queryFn: () =>
      api<CostEstimateOut>(`/api/books/${bookId}/cost-estimate?mode=${mode}${chapterParams(chapters)}`),
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

export function useMintQuote(bookId: string) {
  return useMutation({
    mutationFn: ({ mode, chapters }: { mode: RenderMode; chapters: number[] }) =>
      postJson<QuoteResponse>(`/api/books/${bookId}/quotes`, {
        mode,
        chapters,
        ...(mode === "single" ? { single: {} } : {}),
      }),
  });
}

export function useAttribute(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    // chapters: [] = whole book; a subset merges into the existing report, so a big
    // book can be attributed in installments just like it's rendered
    mutationFn: (chapters: number[]) => postJson<JobOut>(`/api/books/${bookId}/attribute`, { chapters }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["book", bookId] });
    },
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
    queryFn: () => api<SegmentBrowserOut>(`/api/books/${bookId}/chapters/${chapter}/segments`),
    enabled: bookId !== null && attributed,
  });
}

export function useEditLog(bookId: string | null) {
  return useQuery({
    queryKey: ["edits", bookId],
    queryFn: () => api<EditLog>(`/api/books/${bookId}/edits`),
    enabled: bookId !== null,
  });
}

function invalidateReview(qc: ReturnType<typeof useQueryClient>, bookId: string) {
  for (const key of ["characters", "segments", "edits", "estimate"] as const) {
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
}

export function useDraftAssignment(bookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => postJson<AssignmentDraftResponse>(`/api/books/${bookId}/assignment/draft`, {}),
    onSuccess: () => invalidateCasting(qc, bookId),
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
