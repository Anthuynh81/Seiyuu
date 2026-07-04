export function Placeholder({ title, section }: { title: string; section: string }) {
  return (
    <section className="screen">
      <h1>{title}</h1>
      <p className="sub">
        This screen arrives in {section}. The approved design for it lives in{" "}
        <span className="mono">design/m6c/mockup-talkback-v3.html</span>.
      </p>
      <div className="panel">
        <div className="panel-sub">under construction — the console keeps running; jobs and the transport bar are live.</div>
      </div>
    </section>
  );
}
