import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { Route, Routes } from "react-router-dom";

import { useLiveJobs } from "./api/hooks";
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
  useEffect(() => {
    if (prev.current !== undefined && count !== undefined && count !== prev.current) {
      qc.invalidateQueries({ queryKey: ["books"] });
      qc.invalidateQueries({ queryKey: ["book"] });
      if (count < prev.current) {
        for (const key of ["segments", "render-summary", "validation", "estimate", "assignment", "voices"]) {
          qc.invalidateQueries({ queryKey: [key] });
        }
      }
    }
    if (count !== undefined) prev.current = count;
  }, [count, qc]);
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
    </div>
  );
}
