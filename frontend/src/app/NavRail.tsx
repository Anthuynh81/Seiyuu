import { NavLink } from "react-router-dom";

import { useLiveJobs } from "../api/hooks";
import { SCREENS } from "../lib/glossary";
import { useCurrentBook, withBook } from "./currentBook";
import { HealthPill } from "./SystemDrawer";
import { type ThemePref, useTheme } from "./useTheme";

/** The rail reads in workflow order: the shelf, then one book's numbered steps, then the
    tools that serve every book. Routes are unchanged; only labels, order and grouping are. */
const steps = [
  { to: "/review", n: 1, label: SCREENS.review },
  { to: "/lexicon", n: 2, label: SCREENS.lexicon },
  { to: "/render", n: 3, label: SCREENS.render },
  { to: "/listen", n: 4, label: SCREENS.listen },
];

const tools = [
  { to: "/voices", label: SCREENS.voices },
  { to: "/series", label: SCREENS.series },
];

const themes: { id: ThemePref; label: string; title: string }[] = [
  { id: "booth", label: "dark", title: "Dark theme" },
  { id: "daylight", label: "light", title: "Light theme" },
  { id: "system", label: "auto", title: "Follow your computer's setting" },
];

const linkClass = ({ isActive }: { isActive: boolean }) => (isActive ? "on" : "");

export function NavRail() {
  const live = useLiveJobs();
  const { pref, setPref } = useTheme();
  const { bookId, title } = useCurrentBook();
  const running = live.data?.jobs.some((j) => j.state === "running") ?? false;
  return (
    <nav className="nav">
      <div className="brand">
        <b>Seiyuu</b>
        <span>audiobook console</span>
      </div>
      <NavLink to="/" end className={linkClass}>
        {SCREENS.library}
      </NavLink>
      <div role="group" aria-labelledby="nav-steps" className="navgroup">
        <div className="navcap" id="nav-steps">
          Book steps
        </div>
        {title && (
          <div className="navbook" title={title}>
            {title}
          </div>
        )}
        {steps.map((s) => (
          <NavLink key={s.to} to={withBook(s.to, bookId)} className={linkClass}>
            <span className="stepn">{s.n}</span> {s.label}
            {s.to === "/render" && running && <i className="lamp" title="a job is running" />}
          </NavLink>
        ))}
      </div>
      <div role="group" aria-labelledby="nav-tools" className="navgroup tools">
        <div className="navcap" id="nav-tools">
          Tools
        </div>
        {tools.map((t) => (
          <NavLink key={t.to} to={t.to} className={linkClass}>
            {t.label}
          </NavLink>
        ))}
      </div>
      <div className="themesw">
        <span className="cap">theme</span>
        {themes.map((t) => (
          <button key={t.id} className={pref === t.id ? "on" : ""} title={t.title} onClick={() => setPref(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="foot">
        <HealthPill />
        <div>api /api → 127.0.0.1:8000</div>
      </div>
    </nav>
  );
}
