import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { Route, Routes } from "react-router-dom";

import { fetchDepartedJobs, recentJobsQuery, useLiveJobs } from "./api/hooks";
import { JobNotices } from "./app/JobNotices";
import { NavRail } from "./app/NavRail";
import { TransportBar } from "./app/TransportBar";
import { Lexicon } from "./screens/Lexicon";
import { Library } from "./screens/Library";
import { Listen } from "./screens/Listen";
import { RenderJobs } from "./screens/RenderJobs";
import { Review } from "./screens/Review";
import { Series } from "./screens/Series";
import { Voices } from "./screens/Voices";

/** When a job finishes (the live count drops), every screen showing stage artifacts is
    stale — a completed render means new segment timings, a new manifest, new audio.
    Any count transition (start OR finish) refreshes the book payloads: the shelf embeds
    grouped live-job info and the stage flags flip on completion. This watcher is the ONLY
    freshness mechanism for ["books"]/["book"] — their keys are deliberately stable so
    data never blanks mid-session (a blank books.data used to null the derived bookId and
    wipe unsaved casting/lexicon edits). */
function JobCompletionWatcher() {
  const qc = useQueryClient();
  const live = useLiveJobs();
  const count = live.data?.jobs.length;
  const prev = useRef<number | undefined>(undefined);
  const liveIds = live.data?.jobs.map((j) => j.job_id).join("|");
  const prevIds = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (prev.current !== undefined && count !== undefined && count !== prev.current) {
      qc.invalidateQueries({ queryKey: ["books"] });
      qc.invalidateQueries({ queryKey: ["book"] });
      if (count < prev.current) {
        // "characters": a finished speaker re-run changes the roster and which model found it
        const finished = [
          "segments",
          "render-summary",
          "validation",
          "estimate",
          "assignment",
          "voices",
          "characters",
        ];
        for (const key of finished) {
          qc.invalidateQueries({ queryKey: [key] });
        }
        // a finished warm-up flips weights_cached/resident; any finish may follow a setup fix
        qc.invalidateQueries({ queryKey: ["engines"] });
        qc.invalidateQueries({ queryKey: ["system", "probe"] });
      }
    }
    if (count !== undefined) prev.current = count;
  }, [count, qc]);
  // Finish notices + the Library failure banners: when any live job LEAVES the poll (a count
  // drop, or a finish hidden by a new job queuing in the same tick), fetch each departed job
  // by id (it may be older than the 10 newest: the worker runs oldest-first), plus the
  // recent-jobs window (jobs that started and finished inside one poll), and refresh the
  // shelf's job window. New keys only; ["books"]/["book"] untouched.
  useEffect(() => {
    if (liveIds === undefined) return;
    const before = prevIds.current;
    prevIds.current = liveIds;
    if (before === undefined || before === "") return;
    const now = new Set(liveIds === "" ? [] : liveIds.split("|"));
    const departed = before.split("|").filter((id) => !now.has(id));
    if (departed.length > 0) {
      void fetchDepartedJobs(qc, departed);
      qc.fetchQuery({ ...recentJobsQuery, staleTime: 0 }).catch(() => {
        // a failed fetch only delays the notice to the next window refresh
      });
      qc.invalidateQueries({ queryKey: ["jobs", "shelf"] });
    }
  }, [liveIds, qc]);
  return null;
}

export default function App() {
  return (
    <div className="shell">
      <JobCompletionWatcher />
      <NavRail />
      <main className="main">
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/listen" element={<Listen />} />
          <Route path="/review" element={<Review />} />
          <Route path="/lexicon" element={<Lexicon />} />
          <Route path="/voices" element={<Voices />} />
          <Route path="/series" element={<Series />} />
          <Route path="/render" element={<RenderJobs />} />
        </Routes>
      </main>
      <TransportBar />
      {/* last in the DOM so its buttons follow the page in Tab order; it is position:fixed */}
      <JobNotices />
    </div>
  );
}
