import { useCallback, useMemo, useState } from "react";

import { useBooks, useCloudSlots, useVoices } from "../api/hooks";
import { TalkSelect } from "../components/Select";
import { AddVoiceDialog } from "./voices/AddVoiceDialog";
import { CloneDialog } from "./voices/CloneDialog";
import type { DuplicateRecipe } from "./voices/helpers";
import { VoiceCardView } from "./voices/VoiceCardView";

/* -------------------------------------------------- the screen */

type VoiceSort = "name" | "newest" | "kind" | "engine";

export function Voices() {
  const voices = useVoices();
  const slots = useCloudSlots();
  const books = useBooks();
  const [dialog, setDialog] = useState<"none" | "add" | "clone">("none");
  const [duplicate, setDuplicate] = useState<DuplicateRecipe | null>(null);
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<"all" | "preset" | "blend" | "cloned">("all");
  const [tagFilter, setTagFilter] = useState<string | null>(null);
  const [sort, setSort] = useState<VoiceSort>("name");

  // auto-cast tags a voice with the book_id it was cast for — show the title instead
  const titleFor = useCallback(
    (tag: string) => books.data?.books.find((b) => b.book_id === tag)?.title ?? tag,
    [books.data],
  );

  // Memoized: this screen re-renders on every 2s job poll (useBooks), and the tag-union +
  // triple filter + sort chains over the whole library are pure functions of these inputs.
  const all = useMemo(() => voices.data?.voices ?? [], [voices.data]);
  const allTags = useMemo(
    () => [...new Set(all.flatMap((v) => v.tags))].sort((a, b) => titleFor(a).localeCompare(titleFor(b))),
    [all, titleFor],
  );
  const q = query.trim().toLowerCase();
  const shown = useMemo(
    () =>
      all
        .filter((v) => kindFilter === "all" || v.kind === kindFilter)
        .filter((v) => tagFilter === null || v.tags.includes(tagFilter))
        .filter(
          (v) =>
            !q ||
            v.name.toLowerCase().includes(q) ||
            v.voice_id.toLowerCase().includes(q) ||
            v.tags.some((t) => titleFor(t).toLowerCase().includes(q)),
        )
        .sort((a, b) => {
          switch (sort) {
            case "newest":
              return b.created_at.localeCompare(a.created_at) || a.name.localeCompare(b.name);
            case "kind":
              return a.kind.localeCompare(b.kind) || a.name.localeCompare(b.name);
            case "engine":
              return a.engine.localeCompare(b.engine) || a.name.localeCompare(b.name);
            default:
              return a.name.localeCompare(b.name);
          }
        }),
    [all, kindFilter, tagFilter, q, sort, titleFor],
  );

  return (
    <section className="screen">
      <h1>Voice Studio</h1>
      <p className="sub">
        Presets, blends, and consent-attested clones. Auditions are live synthesis — the console refuses with a reason
        when the GPU or the cloud is otherwise engaged.
      </p>
      <div className="panel">
        <div className="panel-h">
          <b>Voices</b>
          <span className="tag ml-3.5">{voices.data ? `${voices.data.voices.length} in library` : "…"}</span>
          <div className="slotbank" title="ElevenLabs voice slots">
            <span className="tag mr-1.5">
              cloud slots {slots.data ? `${slots.data.count}/${slots.data.max_slots}` : "…"}
            </span>
            {Array.from({ length: slots.data?.max_slots ?? 10 }, (_, i) => (
              <i key={i} className={i < (slots.data?.count ?? 0) ? "lit" : ""} />
            ))}
          </div>
          <button
            className="key quiet ml-3.5"
            onClick={() => {
              setDuplicate(null);
              setDialog("add");
            }}
          >
            add voice
          </button>
          <button className="key ml-2" onClick={() => setDialog("clone")}>new clone</button>
        </div>
        <div className="voicetools">
          <input
            type="search"
            className="taginput w-[200px] flex-none"
            placeholder="search name, id, tag…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="search voices"
          />
          {(["all", "preset", "blend", "cloned"] as const).map((k) => (
            <button key={k} className={`chap ${kindFilter === k ? "on" : ""}`} onClick={() => setKindFilter(k)}>
              {k}
            </button>
          ))}
          <span className="flex-1" />
          <span className="tag">sort</span>
          <TalkSelect
            ariaLabel="sort voices"
            value={sort}
            onChange={(v) => setSort(v as VoiceSort)}
            options={[
              { value: "name", label: "name" },
              { value: "newest", label: "newest" },
              { value: "kind", label: "kind" },
              { value: "engine", label: "engine" },
            ]}
          />
        </div>
        {allTags.length > 0 && (
          <div className="voicetools pt-0">
            <span className="tag">tags</span>
            <button className={`tagbtn ${tagFilter === null ? "on" : ""}`} onClick={() => setTagFilter(null)}>
              all
            </button>
            {allTags.map((t) => (
              <button
                key={t}
                className={`tagbtn ${tagFilter === t ? "on" : ""}`}
                title={t}
                onClick={() => setTagFilter(tagFilter === t ? null : t)}
              >
                {titleFor(t)}
              </button>
            ))}
          </div>
        )}
        {voices.isPending && <div className="loadline p-3.5">opening the booth…</div>}
        {voices.isError && <div className="errline m-3.5">{voices.error.message}</div>}
        {voices.data?.unreadable.map((u) => (
          <div className="refusal mx-3.5 mt-2.5" key={u.voice_id}>
            <span className="tag">unreadable</span>
            <p>{u.voice_id}: {u.error}</p>
          </div>
        ))}
        {voices.data && voices.data.voices.length === 0 && (
          <div className="loadline p-3.5">
            no voices yet — add a preset to get narrating, or clone from a reference clip
          </div>
        )}
        {voices.data && all.length > 0 && shown.length === 0 && (
          <div className="loadline p-3.5">
            nothing matches those filters — {all.length} voice(s) hidden
          </div>
        )}
        <div className="voices p-3.5">
          {shown.map((v) => (
            <VoiceCardView
              key={v.voice_id}
              voice={v}
              titleFor={titleFor}
              onDuplicate={(recipe) => {
                setDuplicate(recipe);
                setDialog("add");
              }}
            />
          ))}
        </div>
      </div>
      {dialog === "add" && (
        <AddVoiceDialog
          initial={duplicate ?? undefined}
          onClose={() => {
            setDialog("none");
            setDuplicate(null);
          }}
        />
      )}
      {dialog === "clone" && <CloneDialog onClose={() => setDialog("none")} />}
    </section>
  );
}
