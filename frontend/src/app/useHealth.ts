import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { useEngines, useLiveJobs, useSystemProbe } from "../api/hooks";
import { buildChecklist, type HealthLevel, overallHealth } from "../lib/health";

/** Ready / Needs setup / Backend offline, derived from the same rows the checklist shows.
    Mounted for the app's lifetime (NavRail), so it also owns the one recovery hop: when the
    probe goes from failing to answering, the engine catalog (which has no poll of its own)
    is refetched once so its rows don't stay missing after the backend comes back. */
export function useHealth(): HealthLevel {
  const qc = useQueryClient();
  const probe = useSystemProbe();
  const engines = useEngines();
  const live = useLiveJobs();

  const wasFailing = useRef(false);
  useEffect(() => {
    if (probe.isError) wasFailing.current = true;
    else if (probe.isSuccess && wasFailing.current) {
      wasFailing.current = false;
      void qc.invalidateQueries({ queryKey: ["engines"] });
    }
  }, [probe.isError, probe.isSuccess, qc]);

  const rows = buildChecklist({
    probe: probe.data,
    probeError: probe.error,
    engines: engines.data?.engines ?? (engines.isError ? null : undefined),
    liveJobs: live.data?.jobs,
  });
  return overallHealth(rows, probe.error, probe.data !== undefined);
}
