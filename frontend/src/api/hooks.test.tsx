import { QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { makeQueryClient, mockApi } from "../test/utils";
import { ApiError } from "./client";
import type { SegmentWordsClip } from "./hooks";
import { useDeleteVoice, useSaveLexicon, useSegmentWords, useSwitchRenderMode } from "./hooks";
import type { LexiconEntry, LexiconSaved, SegmentWords } from "./types";

function createWrapper() {
  const queryClient = makeQueryClient();
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return { queryClient, Wrapper };
}

describe("useSegmentWords", () => {
  const wordsFixture: SegmentWords = {
    words: [
      { start: 0, end: 0.4, word: "Hello" },
      { start: 0.4, end: 0.9, word: "there" },
    ],
    audio_duration: 0.9,
    source: "whisper",
  };

  const clips: SegmentWordsClip[] = [
    { key: "blk-ok:0", blockId: "blk-ok", segment: 0, audioKey: "aud-123" },
    { key: "blk-missing:1", blockId: "blk-missing", segment: 1, audioKey: null },
  ];
  // the queryKey folds every clip's audio identity (sorted): a re-render re-keys
  const clipsSig = "blk-missing:1@|blk-ok:0@aud-123";

  it("puts only server-resolved clips in byKey and folds their audioKey into sig; omitted clips stay absent", async () => {
    const server = mockApi();
    server.get("/api/books/b1/chapters/3/words", {
      book_id: "b1",
      chapter: 3,
      words: { "blk-ok:0": wordsFixture }, // blk-missing omitted: wav missing / not aligned
    });

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useSegmentWords("b1", 3, clips), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.byKey.size).toBe(1));
    expect(result.current.byKey.get("blk-ok:0")).toEqual(wordsFixture);
    expect(result.current.byKey.has("blk-missing:1")).toBe(false);
    // sig folds the resolved clip's audio identity, so a re-render that swaps the wav
    // (same clip key, new audio_key) flips the signature
    expect(result.current.sig).toBe("blk-ok:0@aud-123");
    // ONE batch fetch for the whole chapter — not a request per clip
    expect(server.lastCall("GET", "words")?.url).toBe("/api/books/b1/chapters/3/words");
  });

  it("a chapter-level 404 (not yet rendered) settles as success with no resolved clips", async () => {
    const server = mockApi();
    server.error("GET", "/api/books/b1/chapters/3/words", 404, "not_found", "no rendered audio");

    const { queryClient, Wrapper } = createWrapper();
    const { result } = renderHook(() => useSegmentWords("b1", 3, clips), { wrapper: Wrapper });

    // the 404 is caught and becomes null data — the read-along keeps interpolation
    // instead of erroring the whole page
    await waitFor(() => {
      expect(queryClient.getQueryState(["segment-words", "b1", 3, clipsSig])?.status).toBe("success");
    });
    expect(result.current.byKey.size).toBe(0);
    expect(result.current.sig).toBe("");
  });
});

describe("useDeleteVoice", () => {
  it("rejects with ApiError voice_referenced and never issues the DELETE when references exist", async () => {
    const server = mockApi();
    server.get("/api/voices/v1/references", {
      voice_id: "v1",
      references: [{ book_id: "dune", role: "Paul" }],
    });
    const { Wrapper } = createWrapper();
    const { result } = renderHook(useDeleteVoice, { wrapper: Wrapper });

    let err: unknown;
    await act(async () => {
      err = await result.current.mutateAsync("v1").catch((e: unknown) => e);
    });
    expect(err).toBeInstanceOf(ApiError);
    const apiErr = err as ApiError;
    expect(apiErr.code).toBe("voice_referenced");
    expect(apiErr.status).toBe(409);
    expect(apiErr.message).toContain("dune (Paul)");
    expect(server.calls.filter((c) => c.method === "DELETE")).toHaveLength(0);
  });

  it("DELETEs and invalidates the voices query when the voice is unreferenced", async () => {
    const server = mockApi();
    server.get("/api/voices/v2/references", { voice_id: "v2", references: [] });
    server.delete("/api/voices/v2", { deleted: "v2" });
    const { queryClient, Wrapper } = createWrapper();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const { result } = renderHook(useDeleteVoice, { wrapper: Wrapper });

    let out: { deleted: string } | undefined;
    await act(async () => {
      out = await result.current.mutateAsync("v2");
    });
    expect(out).toEqual({ deleted: "v2" });
    expect(server.lastCall("DELETE", "/api/voices/v2")).toBeDefined();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["voices"] });
  });
});

describe("useSwitchRenderMode", () => {
  it("POSTs the mode and invalidates every reader of the active manifest — including the exact keys Listen mounts", async () => {
    const server = mockApi();
    server.post("/api/books/b1/render/mode", { book_id: "b1", active_mode: "single" });
    const { queryClient, Wrapper } = createWrapper();
    // seed the precise key shapes Listen uses (useRenderSummary folds `rendered`, useSegments
    // folds the chapter) so this fails if the prefix invalidation ever stops covering them
    queryClient.setQueryData(["render-summary", "b1", true], { active_mode: "multi" });
    queryClient.setQueryData(["segments", "b1", 1], { segments: [] });
    const { result } = renderHook(() => useSwitchRenderMode("b1"), { wrapper: Wrapper });

    await act(async () => {
      await result.current.mutateAsync("single");
    });

    expect(server.jsonBodyOf("POST", "/api/books/b1/render/mode")).toEqual({ mode: "single" });
    expect(queryClient.getQueryState(["render-summary", "b1", true])?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(["segments", "b1", 1])?.isInvalidated).toBe(true);
  });
});

describe("useSaveLexicon", () => {
  it("PUTs the entries and invalidates the book's lexicon and estimate on success", async () => {
    const entries: LexiconEntry[] = [
      { term: "Seiyuu", respelling: "SAY-yoo", ipa: null, note: null, case_sensitive: false },
    ];
    const saved: LexiconSaved = {
      book_id: "b1",
      schema_version: 1,
      entries,
      affected_blocks: 4,
      total_speakable_blocks: 90,
    };
    const server = mockApi();
    server.put("/api/books/b1/lexicon", saved);
    const { queryClient, Wrapper } = createWrapper();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const { result } = renderHook(() => useSaveLexicon("b1"), { wrapper: Wrapper });

    let out: LexiconSaved | undefined;
    await act(async () => {
      out = await result.current.mutateAsync(entries);
    });
    expect(out?.affected_blocks).toBe(4);

    const put = server.lastCall("PUT", "/api/books/b1/lexicon");
    expect(put).toBeDefined();
    expect(server.jsonBodyOf("PUT", "/api/books/b1/lexicon")).toEqual({ entries });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["lexicon", "b1"] });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["estimate", "b1"] });
  });
});
