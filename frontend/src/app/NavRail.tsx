import { NavLink } from "react-router-dom";

import { useLiveJobs } from "../api/hooks";
import { type ThemePref, useTheme } from "./useTheme";

const items = [
  { to: "/", label: "Library" },
  { to: "/listen", label: "Listen" },
  { to: "/review", label: "Character Review" },
  { to: "/lexicon", label: "Pronunciation" },
  { to: "/voices", label: "Voice Studio" },
  { to: "/series", label: "Series" },
  { to: "/render", label: "Render & Jobs" },
];

const themes: { id: ThemePref; label: string; title: string }[] = [
  { id: "booth", label: "booth", title: "dark — the booth" },
  { id: "daylight", label: "day", title: "light — the daylight desk" },
  { id: "system", label: "auto", title: "follow the system" },
];

export function NavRail() {
  const live = useLiveJobs();
  const { pref, setPref } = useTheme();
  const running = live.data?.jobs.some((j) => j.state === "running") ?? false;
  return (
    <nav className="nav">
      <div className="brand">
        <b>Seiyuu</b>
        <span>audiobook console</span>
      </div>
      {items.map((item) => (
        <NavLink key={item.to} to={item.to} end={item.to === "/"} className={({ isActive }) => (isActive ? "on" : "")}>
          {item.label}
          {item.to === "/render" && running && <i className="lamp" title="a job is running" />}
        </NavLink>
      ))}
      <div className="themesw">
        <span className="cap">lights</span>
        {themes.map((t) => (
          <button key={t.id} className={pref === t.id ? "on" : ""} title={t.title} onClick={() => setPref(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="foot">
        api /api → 127.0.0.1:8000
        <br />
        talkback v4 · console online
      </div>
    </nav>
  );
}
