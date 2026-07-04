import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, postForm } from "./client";
import type { BooksOut, IngestResponse, JobOut, JobsOut } from "./types";

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
