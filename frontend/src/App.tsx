import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { Route, Routes } from "react-router-dom";

import { useLiveJobs } from "./api/hooks";
import { NavRail } from "./app/NavRail";
import { TransportBar } from "./app/TransportBar";
import { Library } from "./screens/Library";
import { Listen } from "./screens/Listen";
import { RenderJobs } from "./screens/RenderJobs";
import { Review } from "./screens/Review";
import { Voices } from "./screens/Voices";

/** When a job finishes (the live count drops), every screen showing stage artifacts is
    stale — a completed render means new segment timings, a new manifest, new audio. */
function JobCompletionWatcher() {
  const qc = useQueryClient();
  const live = useLiveJobs();
  const count = live.data?.jobs.length;
  const prev = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (prev.current !== undefined && count !== undefined && count < prev.current) {
      for (const key of ["segments", "render-summary", "validation", "estimate", "assignment", "voices"]) {
        qc.invalidateQueries({ queryKey: [key] });
      }
    }
    prev.current = count;
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
          <Route path="/voices" element={<Voices />} />
          <Route path="/render" element={<RenderJobs />} />
        </Routes>
      </main>
      <TransportBar />
    </div>
  );
}
