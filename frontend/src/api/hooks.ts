import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, postForm, postJson } from "./client";
import type {
  BookDetail,
  BooksOut,
  CostEstimateOut,
  IngestResponse,
  JobOut,
  JobsOut,
  QuoteResponse,
  RenderMode,
  RenderRequest,
  ValidationReportOut,
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

export function useEstimate(bookId: string | null, mode: RenderMode, ready: boolean) {
  return useQuery({
    queryKey: ["estimate", bookId, mode],
    queryFn: () => api<CostEstimateOut>(`/api/books/${bookId}/cost-estimate?mode=${mode}`),
    enabled: bookId !== null && ready,
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
    mutationFn: (mode: RenderMode) =>
      postJson<QuoteResponse>(`/api/books/${bookId}/quotes`, {
        mode,
        ...(mode === "single" ? { single: {} } : {}),
      }),
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
