import { useEffect, useState } from "react";

/** The two rooms: booth (dark, primary) and daylight desk (light); system follows the OS.
    index.html stamps data-theme before first paint from the same localStorage key, so this
    hook only has to keep it current after hydration. */
export type ThemePref = "booth" | "daylight" | "system";

const KEY = "seiyuu-theme";

function readPref(): ThemePref {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw === "booth" || raw === "daylight" || raw === "system") return raw;
  } catch {
    /* private mode / storage denied — fall through */
  }
  return "system";
}

export function useTheme(): { pref: ThemePref; setPref: (p: ThemePref) => void } {
  const [pref, setPref] = useState<ThemePref>(readPref);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, pref);
    } catch {
      /* non-persistent is fine — the stamp below still applies */
    }
    const mq = matchMedia("(prefers-color-scheme: light)");
    const apply = () => {
      const light = pref === "daylight" || (pref === "system" && mq.matches);
      document.documentElement.dataset.theme = light ? "light" : "dark";
    };
    apply();
    if (pref === "system") {
      mq.addEventListener("change", apply);
      return () => mq.removeEventListener("change", apply);
    }
  }, [pref]);

  return { pref, setPref };
}
