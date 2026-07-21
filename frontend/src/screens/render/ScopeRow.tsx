import type { ChapterSummary } from "../../api/types";
import { Tip } from "../../components/Tooltip";
import { continueRange, scopeChapters, type Scope } from "../../lib/scope";

/* -------------------------------------------------- chapter scope */

export function ScopeRow({
  scope,
  setScope,
  chapters,
  renderedSet,
}: {
  scope: Scope;
  setScope: (s: Scope) => void;
  chapters: ChapterSummary[];
  renderedSet: Set<number>;
}) {
  const total = chapters.length;
  const cont = continueRange(renderedSet, total, 10);
  const selected = scopeChapters(scope, total);
  const speakable = selected.length
    ? chapters.filter((c) => selected.includes(c.index)).reduce((a, c) => a + c.speakable_blocks, 0)
    : chapters.reduce((a, c) => a + c.speakable_blocks, 0);

  return (
    <div className="scoperow">
      <span className="tag">scope</span>
      <button className={`chap ${scope.kind === "whole" ? "on" : ""}`} onClick={() => setScope({ kind: "whole" })}>
        whole book
      </button>
      <button
        className={`chap ${scope.kind === "range" ? "on" : ""}`}
        onClick={() => setScope(cont ?? { kind: "range", from: 1, to: Math.min(10, total) })}
      >
        chapter range
      </button>
      {scope.kind === "range" && (
        <>
          <label className="rangelbl">
            ch
            <input
              type="number"
              min={1}
              max={total}
              value={scope.from}
              onChange={(e) => setScope({ ...scope, from: Number(e.target.value) || 1 })}
            />
          </label>
          <label className="rangelbl">
            to
            <input
              type="number"
              min={1}
              max={total}
              value={scope.to}
              onChange={(e) => setScope({ ...scope, to: Number(e.target.value) || scope.from })}
            />
          </label>
          {cont && (
            <Tip content="the next ten chapters without rendered audio">
              <button className="key quiet px-[9px] py-[3px]" onClick={() => setScope(cont)}>
                continue · next 10 from ch {cont.from}
              </button>
            </Tip>
          )}
        </>
      )}
      <span className="mono scopehint">
        {selected.length ? `${selected.length} chapter(s)` : `all ${total} chapters`} · {speakable.toLocaleString()}{" "}
        segments
        {renderedSet.size > 0 && ` · ${renderedSet.size} ch already rendered`}
      </span>
    </div>
  );
}
