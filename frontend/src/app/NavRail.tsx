import { NavLink } from "react-router-dom";

import { useLiveJobs } from "../api/hooks";

const items = [
  { to: "/", label: "Library" },
  { to: "/listen", label: "Listen" },
  { to: "/review", label: "Character Review" },
  { to: "/lexicon", label: "Pronunciation" },
  { to: "/voices", label: "Voice Studio" },
  { to: "/series", label: "Series" },
  { to: "/render", label: "Render & Jobs" },
];

export function NavRail() {
  const live = useLiveJobs();
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
      <div className="foot">
        api /api → 127.0.0.1:8000
        <br />
        m6c-1 · library online
      </div>
    </nav>
  );
}
