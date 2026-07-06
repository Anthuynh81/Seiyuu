import { useEffect, useRef, useState } from "react";

import { ApiError } from "../api/client";
import {
  useAudition,
  useBooks,
  useCloneVoice,
  useCloudSlots,
  useCreateVoice,
  useDeleteVoice,
  useKokoroPresets,
  useSetVoiceTags,
  useVoices,
  useWarmup,
} from "../api/hooks";
import type { VoiceCreate, VoiceOut } from "../api/types";

/* -------------------------------------------------- audition control */

const linkBtnStyle = { background: "none", border: "none", color: "var(--tungsten)", cursor: "pointer", padding: 0 } as const;
const BORROW_RETRY_MAX = 3; // bounded auto-retries while a render lends the GPU between segments

function AuditionControl({ voice }: { voice: VoiceOut }) {
  const audition = useAudition(voice.voice_id);
  const warmup = useWarmup();
  const [playerOpen, setPlayerOpen] = useState(false);
  const err = audition.error instanceof ApiError ? audition.error : null;

  // gpu_busy_retry is a SOFT refusal: a render is lending the GPU between segments, so the
  // right move is to wait and retry — not to give up. Auto-retry with a short backoff, up to
  // BORROW_RETRY_MAX, preserving the same confirm_paid the user chose.
  const borrowRetries = useRef(0);
  useEffect(() => {
    if (audition.isSuccess) borrowRetries.current = 0;
  }, [audition.isSuccess]);
  useEffect(() => {
    if (err?.code !== "gpu_busy_retry" || borrowRetries.current >= BORROW_RETRY_MAX) return;
    borrowRetries.current += 1;
    const confirmPaid = audition.variables ?? false;
    const timer = setTimeout(() => audition.mutate(confirmPaid), 500 * borrowRetries.current);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [err]);
  const startAudition = (confirmPaid: boolean) => {
    borrowRetries.current = 0; // a fresh user-initiated attempt resets the borrow budget
    audition.mutate(confirmPaid);
  };

  if (audition.isPending) {
    return (
      <div className="audit">
        <span className="stop" />
        <span className="lbl">auditioning…</span>
        <span className="live" />
      </div>
    );
  }

  // soft borrow-retry: while attempts remain, show a "queued behind a render" spinner rather
  // than a refusal — the render keeps running and the audition slips in between its segments
  if (err?.code === "gpu_busy_retry" && borrowRetries.current < BORROW_RETRY_MAX) {
    return (
      <div className="audit" title="a render is lending the GPU between segments — it keeps running">
        <span className="stop" />
        <span className="lbl">queued behind a render segment — retrying…</span>
        <span className="live" />
      </div>
    );
  }

  if (err) {
    const detail = (err.detail ?? {}) as Record<string, unknown>;
    const recourse = (() => {
      switch (err.code) {
        case "engine_cold":
          return (
            <button
              className="link"
              style={linkBtnStyle}
              disabled={warmup.isPending}
              onClick={() => warmup.mutate(voice.engine, { onSuccess: () => audition.reset() })}
            >
              {warmup.isPending ? "starting warmup…" : "warm up first"}
            </button>
          );
        case "payment_confirmation_required":
          return (
            <button className="link" style={linkBtnStyle} onClick={() => startAudition(true)}>
              confirm ~${Number(detail.estimated_usd ?? 0).toFixed(4)} &amp; play
            </button>
          );
        case "gpu_busy_retry":
          // auto-retries exhausted — the render is still holding the GPU; let the user retry
          return (
            <span>
              the render keeps running — it hasn't yielded the GPU yet;{" "}
              <button className="link" style={linkBtnStyle} onClick={() => startAudition(audition.variables ?? false)}>
                retry
              </button>
            </span>
          );
        case "gpu_busy":
        case "cloud_busy":
          return <span>wait for the job in the transport bar, or cancel it — <button className="link" style={linkBtnStyle} onClick={() => audition.reset()}>retry</button></span>;
        default:
          return (
            <button className="link" style={linkBtnStyle} onClick={() => audition.reset()}>
              dismiss
            </button>
          );
      }
    })();
    return (
      <div className="refusal">
        <span className="tag">{err.code}</span>
        <p>
          {err.message} — {recourse}
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="audit" role="button" tabIndex={0} onClick={() => startAudition(false)} style={{ cursor: "pointer" }}>
        <span className="play" />
        <span className="lbl">audition</span>
        {voice.has_audition && (
          <button
            className="key quiet"
            style={{ marginLeft: "auto", padding: "1px 8px", fontSize: 10.5 }}
            onClick={(e) => {
              e.stopPropagation();
              setPlayerOpen(!playerOpen);
            }}
          >
            {playerOpen ? "hide last" : "▶ last take"}
          </button>
        )}
      </div>
      {(playerOpen || audition.isSuccess) && voice.has_audition && (
        <audio
          controls
          autoPlay={audition.isSuccess}
          src={`/api/voices/${voice.voice_id}/audition.wav?t=${audition.isSuccess ? Date.now() : 0}`}
          style={{ width: "100%", marginTop: 8, height: 30 }}
        />
      )}
    </div>
  );
}

/* -------------------------------------------------- tags */

function TagEditor({ voice, titleFor }: { voice: VoiceOut; titleFor: (tag: string) => string }) {
  const setTags = useSetVoiceTags();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const save = () =>
    setTags.mutate(
      { voiceId: voice.voice_id, tags: draft.split(",").map((s) => s.trim()).filter(Boolean) },
      { onSuccess: () => setEditing(false) },
    );
  if (!editing) {
    return (
      <div className="tagsrow">
        {voice.tags.map((t) => (
          <span key={t} className="tagchip" title={t}>{titleFor(t)}</span>
        ))}
        <button
          className="rowedit"
          style={{ visibility: "visible", marginLeft: 0 }}
          title="edit tags"
          onClick={() => {
            setDraft(voice.tags.join(", "));
            setEditing(true);
          }}
        >
          {voice.tags.length ? "✎" : "+ tag"}
        </button>
      </div>
    );
  }
  return (
    <div className="tagsrow">
      <input
        className="taginput"
        value={draft}
        autoFocus
        placeholder="comma, separated, tags"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") setEditing(false);
        }}
      />
      <button className="key quiet" style={{ padding: "2px 8px" }} disabled={setTags.isPending} onClick={save}>
        save
      </button>
      {setTags.error && (
        <span className="mono" style={{ color: "var(--clip)", fontSize: 10 }}>{setTags.error.message}</span>
      )}
    </div>
  );
}

/* -------------------------------------------------- voice card */

function VoiceCardView({ voice, titleFor }: { voice: VoiceOut; titleFor: (tag: string) => string }) {
  const del = useDeleteVoice();
  const kindLine =
    voice.kind === "preset"
      ? `${voice.preset_id} · seed ${voice.seed}`
      : voice.kind === "blend"
        ? (voice.blend ?? []).map((b) => `${b.preset_id} ${b.weight}`).join(" : ")
        : `${voice.reference_audio ?? "reference.wav"} · seed ${voice.seed}`;
  const delErr = del.error instanceof ApiError ? del.error : null;
  return (
    <div className="vcard">
      <span className="eng">
        {voice.engine} · {voice.kind}
        <button
          className="rowedit"
          style={{ float: "right", visibility: "visible" }}
          title="delete voice"
          disabled={del.isPending}
          onClick={() => del.mutate(voice.voice_id)}
        >
          ✕
        </button>
      </span>
      <h3>{voice.name}</h3>
      <span className="kind">{kindLine}</span>
      {voice.kind === "cloned" ? (
        voice.consent_attested ? (
          <span className="consent">
            <i className="led ok" />
            consent: attested{voice.consent && ` · ${voice.consent.attested_at.slice(0, 10)}`}
          </span>
        ) : (
          <span className="consent">
            <i className="led warn" />
            consent: not attested
          </span>
        )
      ) : (
        <span className="consent" style={{ visibility: "hidden" }}>
          .
        </span>
      )}
      <TagEditor voice={voice} titleFor={titleFor} />
      {delErr && (
        <div className="refusal">
          <span className="tag">{delErr.code}</span>
          <p>{delErr.message}</p>
        </div>
      )}
      <div className="foot">
        <AuditionControl voice={voice} />
      </div>
    </div>
  );
}

/* -------------------------------------------------- preview demos (the mixer's ear) */

/** Play a kokoro preview. Fetch first — refusals (gpu_busy, engine_cold…) come back as
    JSON envelopes, which an <audio src> would swallow silently. */
function useDemoPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null); // the object URL currently backing audioRef
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false); // waiting out a render's GPU borrow

  // pause the prior element and revoke its blob URL — object URLs live until revoked, so tuning
  // a blend (a preview per click) would otherwise leak a blob every click.
  const release = () => {
    audioRef.current?.pause();
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
  };
  useEffect(() => release, []); // on unmount: stop playback and revoke any outstanding url

  const play = async (url: string, attempt = 0) => {
    setError(null);
    audioRef.current?.pause();
    setBusy(url);
    try {
      const res = await fetch(url);
      if (!res.ok) {
        const body = (await res.json().catch(() => null)) as { error?: { code?: string; message?: string } } | null;
        // gpu_busy_retry is soft: a render is lending the GPU between segments — wait & retry
        if (body?.error?.code === "gpu_busy_retry" && attempt < BORROW_RETRY_MAX) {
          setRetrying(true);
          await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
          return play(url, attempt + 1);
        }
        throw new Error(body?.error?.message ?? `preview failed (${res.status})`);
      }
      setRetrying(false);
      const objectUrl = URL.createObjectURL(await res.blob());
      release(); // free the previous preview's element + url now that this one is ready
      urlRef.current = objectUrl;
      const el = new Audio(objectUrl);
      audioRef.current = el;
      const done = () => {
        setBusy(null);
        // revoke only if we're still the current preview (a newer play() may have taken over)
        if (urlRef.current === objectUrl) {
          URL.revokeObjectURL(objectUrl);
          urlRef.current = null;
        }
      };
      el.onended = done;
      el.onerror = done;
      await el.play();
    } catch (e) {
      setBusy(null);
      setRetrying(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  return { play, busy, error, retrying };
}

const presetPreviewUrl = (id: string) => `/api/engines/kokoro/preview?preset=${id}`;
const mixPreviewUrl = (layers: { preset_id: string; weight: number }[]) =>
  `/api/engines/kokoro/preview?components=${layers.map((l) => `${l.preset_id}:${l.weight}`).join(",")}`;

/* -------------------------------------------------- add / clone dialogs */

function AddVoiceDialog({ onClose }: { onClose: () => void }) {
  const create = useCreateVoice();
  const presets = useKokoroPresets();
  const [kind, setKind] = useState<"preset" | "blend">("preset");
  const [name, setName] = useState("");
  const [engine, setEngine] = useState("kokoro");
  const [presetId, setPresetId] = useState("af_heart");
  const [cloudId, setCloudId] = useState("");
  const [gender, setGender] = useState("");
  const [accent, setAccent] = useState<"a" | "b">("a");
  // manual mix: layers of (preset, weight) — the server normalizes weights, so only
  // the ratios matter; the mixer shows the resulting percentages live
  const [manual, setManual] = useState(false);
  const [layers, setLayers] = useState<{ preset_id: string; weight: number }[]>([
    { preset_id: "af_heart", weight: 60 },
    { preset_id: "af_nicole", weight: 40 },
  ]);
  const error = create.error instanceof ApiError ? create.error.message : create.error?.message;

  const demo = useDemoPlayer();
  const catalog =
    presets.data?.voices ??
    [{ id: "af_heart", name: "Heart", language: "en-US", gender: "female", description: null }];
  const describe = (id: string) => catalog.find((p) => p.id === id)?.description;
  const presetOptions = catalog.map((p) => (
    <option key={p.id} value={p.id}>
      {p.id} — {p.gender ?? "?"} {p.language ?? ""}{p.description ? ` · ${p.description}` : ""}
    </option>
  ));
  const demoKey = (url: string, label = "▶") => (
    <button
      className="key quiet"
      style={{ padding: "3px 9px" }}
      disabled={demo.busy !== null}
      title="hear this on the standard audition line"
      onClick={() => demo.play(url)}
    >
      {demo.busy === url ? (demo.retrying ? "queued behind render…" : "playing…") : label}
    </button>
  );
  const totalWeight = layers.reduce((a, l) => a + l.weight, 0) || 1;
  const setLayer = (i: number, patch: Partial<{ preset_id: string; weight: number }>) =>
    setLayers(layers.map((l, k) => (k === i ? { ...l, ...patch } : l)));

  const submit = () => {
    const body: VoiceCreate =
      kind === "preset"
        ? { kind, name, engine, preset_id: engine === "kokoro" ? presetId : cloudId }
        : manual
          ? { kind, name, components: layers.filter((l) => l.weight > 0) }
          : { kind, name, gender: gender || null, accent };
    create.mutate(body, { onSuccess: onClose });
  };

  const active = layers.filter((l) => l.weight > 0);
  // kokoro can't blend across language families (the id's first letter: af_/am_ = American,
  // bf_/bm_ = British) — catch it at the fader instead of a server refusal
  const familyMix = new Set(active.map((l) => l.preset_id[0])).size > 1;
  const blendInvalid = kind === "blend" && manual && (active.length < 2 || familyMix);

  return (
    <div className="overlay on" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog">
        <div className="dh">
          <b>Add voice</b>
          <button className="key quiet" onClick={onClose}>esc</button>
        </div>
        <div className="db">
          <div className="modewrap" style={{ marginBottom: 4 }}>
            <button className={`chap ${kind === "preset" ? "on" : ""}`} onClick={() => setKind("preset")}>preset</button>
            <button className={`chap ${kind === "blend" ? "on" : ""}`} onClick={() => setKind("blend")}>blend</button>
          </div>
          <label>voice name</label>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Narrator" />
          {kind === "preset" ? (
            <>
              <label>engine</label>
              <select value={engine} onChange={(e) => setEngine(e.target.value)}>
                <option value="kokoro">kokoro — local, free</option>
                <option value="elevenlabs">elevenlabs — cloud stock voice, paid to render</option>
              </select>
              {engine === "kokoro" ? (
                <>
                  <label>preset</label>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <select value={presetId} onChange={(e) => setPresetId(e.target.value)} style={{ flex: 1 }}>
                      {presetOptions}
                    </select>
                    {demoKey(presetPreviewUrl(presetId), "▶ demo")}
                  </div>
                  {describe(presetId) && <div className="voicenote">{describe(presetId)}</div>}
                </>
              ) : (
                <>
                  <label>elevenlabs voice id (from their voice library)</label>
                  <input type="text" value={cloudId} onChange={(e) => setCloudId(e.target.value)} placeholder="EXAVITQu…" />
                </>
              )}
            </>
          ) : (
            <>
              <label>recipe</label>
              <div className="modewrap" style={{ marginBottom: 6 }}>
                <button className={`chap ${!manual ? "on" : ""}`} onClick={() => setManual(false)}>auto — from name</button>
                <button className={`chap ${manual ? "on" : ""}`} onClick={() => setManual(true)}>manual mix</button>
              </div>
              {!manual ? (
                <>
                  <label>gender hint for the recipe (optional)</label>
                  <select value={gender} onChange={(e) => setGender(e.target.value)}>
                    <option value="">unknown</option>
                    <option value="female">female</option>
                    <option value="male">male</option>
                  </select>
                  <label>accent</label>
                  <select value={accent} onChange={(e) => setAccent(e.target.value as "a" | "b")}>
                    <option value="a">American</option>
                    <option value="b">British</option>
                  </select>
                </>
              ) : (
                <>
                  {layers.map((l, i) => (
                    <div key={i}>
                      <div className="mixrow">
                        {demoKey(presetPreviewUrl(l.preset_id))}
                        <select value={l.preset_id} onChange={(e) => setLayer(i, { preset_id: e.target.value })}>
                          {presetOptions}
                        </select>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={l.weight}
                          aria-label={`weight of ${l.preset_id}`}
                          onChange={(e) => setLayer(i, { weight: Number(e.target.value) })}
                        />
                        <span className="pct">{Math.round((100 * l.weight) / totalWeight)}%</span>
                        <button
                          className="rowedit"
                          style={{ visibility: "visible" }}
                          title="remove layer"
                          disabled={layers.length <= 2}
                          onClick={() => setLayers(layers.filter((_, k) => k !== i))}
                        >
                          ✕
                        </button>
                      </div>
                      {describe(l.preset_id) && <div className="voicenote">{describe(l.preset_id)}</div>}
                    </div>
                  ))}
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 6 }}>
                    <button
                      className="key quiet"
                      style={{ padding: "3px 9px" }}
                      onClick={() => setLayers([...layers, { preset_id: catalog[0].id, weight: 30 }])}
                    >
                      + add layer
                    </button>
                    {!blendInvalid && demoKey(mixPreviewUrl(active), "▶ demo mix")}
                    <span className="mono" style={{ fontSize: 10.5, color: "var(--ink-3)" }}>
                      faders are ratios — the mix normalizes itself
                    </span>
                  </div>
                  {blendInvalid && (
                    <div className="mono" style={{ fontSize: 11, color: "var(--caution)", marginTop: 6 }}>
                      {familyMix
                        ? "kokoro can't blend across accents — keep every layer American (a…) or every layer British (b…)"
                        : "a blend needs at least two layers with weight"}
                    </div>
                  )}
                </>
              )}
            </>
          )}
          {demo.error && <div className="errline" style={{ marginTop: 12 }}>{demo.error}</div>}
          {error && <div className="errline" style={{ marginTop: 12 }}>{error}</div>}
        </div>
        <div className="df">
          <button className="key quiet" onClick={onClose}>cancel</button>
          <button
            className="key"
            disabled={
              create.isPending ||
              !name.trim() ||
              blendInvalid ||
              (kind === "preset" && engine === "elevenlabs" && !cloudId.trim())
            }
            onClick={submit}
          >
            add voice
          </button>
        </div>
      </div>
    </div>
  );
}

function CloneDialog({ onClose }: { onClose: () => void }) {
  const clone = useCloneVoice();
  const fileInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [engine, setEngine] = useState("chatterbox");
  const [attested, setAttested] = useState(false);
  const [attestedBy, setAttestedBy] = useState(() => localStorage.getItem("seiyuu.attestedBy") ?? "");
  const err = clone.error instanceof ApiError ? clone.error : null;
  const recloneBlocked = err?.code === "reclone_blocked";

  const submit = (replace = false) => {
    if (!file) return;
    localStorage.setItem("seiyuu.attestedBy", attestedBy);
    clone.mutate({ file, name, engine, attestedBy, replace }, { onSuccess: onClose });
  };

  return (
    <div className="overlay on" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog">
        <div className="dh">
          <b>New cloned voice</b>
          <button className="key quiet" onClick={onClose}>esc</button>
        </div>
        <div className="db">
          <label>reference clip (wav/mp3, ≥ 20 s recommended)</label>
          <div style={{ display: "flex", gap: 8 }}>
            <input type="text" readOnly value={file?.name ?? ""} placeholder="choose a file…" onClick={() => fileInput.current?.click()} />
            <button className="key quiet" onClick={() => fileInput.current?.click()}>browse</button>
            <input ref={fileInput} type="file" accept="audio/*" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <label>voice name</label>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Mr. Darcy" />
          <label>engine</label>
          <select value={engine} onChange={(e) => setEngine(e.target.value)}>
            <option value="chatterbox">chatterbox — local, free</option>
            <option value="elevenlabs">elevenlabs — cloud IVC, paid</option>
          </select>
          <div className="paper release slim">
            <div className="att">
              <input type="checkbox" id="att" checked={attested} onChange={(e) => setAttested(e.target.checked)} />
              <label htmlFor="att" className="attlbl" style={{ margin: 0 }}>
                I have the speaker's permission to clone this voice
              </label>
              <span style={{ marginLeft: "auto", color: "var(--paper-ink-2)", fontFamily: "var(--mono)", fontSize: 11 }}>
                as{" "}
                <input
                  type="text"
                  value={attestedBy}
                  onChange={(e) => setAttestedBy(e.target.value)}
                  placeholder="your name"
                  style={{ background: "transparent", border: "none", borderBottom: "1px solid var(--paper-ink-2)", color: "var(--paper-ink)", fontFamily: "var(--mono)", width: 110, padding: "1px 3px" }}
                />
              </span>
            </div>
          </div>
          <div className="tag" style={{ marginTop: 8, color: "var(--ink-3)", textTransform: "none", letterSpacing: ".02em" }}>
            one click — the attestation binds to these exact bytes (sha-256) and is required by the render gate
          </div>
          {err && !recloneBlocked && <div className="errline" style={{ marginTop: 12 }}>{err.message}</div>}
          {recloneBlocked && (
            <div className="refusal" style={{ marginTop: 12 }}>
              <span className="tag">reclone_blocked</span>
              <p>
                {err!.message} —{" "}
                <button className="link" style={{ background: "none", border: "none", color: "var(--tungsten)", cursor: "pointer", padding: 0 }} onClick={() => submit(true)}>
                  replace it (purges its cached audio; paid segments re-bill)
                </button>
              </p>
            </div>
          )}
        </div>
        <div className="df">
          <button className="key quiet" onClick={onClose}>cancel</button>
          <button className="key" disabled={clone.isPending || !file || !name.trim() || !attested || !attestedBy.trim()} onClick={() => submit(false)}>
            {clone.isPending ? "cloning…" : "clone voice"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------- the screen */

type VoiceSort = "name" | "newest" | "kind" | "engine";

export function Voices() {
  const voices = useVoices();
  const slots = useCloudSlots();
  const books = useBooks();
  const [dialog, setDialog] = useState<"none" | "add" | "clone">("none");
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<"all" | "preset" | "blend" | "cloned">("all");
  const [tagFilter, setTagFilter] = useState<string | null>(null);
  const [sort, setSort] = useState<VoiceSort>("name");

  // auto-cast tags a voice with the book_id it was cast for — show the title instead
  const titleFor = (tag: string) => books.data?.books.find((b) => b.book_id === tag)?.title ?? tag;

  const all = voices.data?.voices ?? [];
  const allTags = [...new Set(all.flatMap((v) => v.tags))].sort((a, b) =>
    titleFor(a).localeCompare(titleFor(b)),
  );
  const q = query.trim().toLowerCase();
  const shown = all
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
    });

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
          <span className="tag" style={{ marginLeft: 14 }}>{voices.data ? `${voices.data.voices.length} in library` : "…"}</span>
          <div className="slotbank" title="ElevenLabs voice slots">
            <span className="tag" style={{ marginRight: 6 }}>
              cloud slots {slots.data ? `${slots.data.count}/${slots.data.max_slots}` : "…"}
            </span>
            {Array.from({ length: slots.data?.max_slots ?? 10 }, (_, i) => (
              <i key={i} className={i < (slots.data?.count ?? 0) ? "lit" : ""} />
            ))}
          </div>
          <button className="key quiet" style={{ marginLeft: 14 }} onClick={() => setDialog("add")}>add voice</button>
          <button className="key" style={{ marginLeft: 8 }} onClick={() => setDialog("clone")}>new clone</button>
        </div>
        <div className="voicetools">
          <input
            type="search"
            className="taginput"
            style={{ width: 200 }}
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
          <span style={{ flex: 1 }} />
          <span className="tag">sort</span>
          <select value={sort} onChange={(e) => setSort(e.target.value as VoiceSort)}>
            <option value="name">name</option>
            <option value="newest">newest</option>
            <option value="kind">kind</option>
            <option value="engine">engine</option>
          </select>
        </div>
        {allTags.length > 0 && (
          <div className="voicetools" style={{ paddingTop: 0 }}>
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
        {voices.isPending && <div className="loadline" style={{ padding: 14 }}>opening the booth…</div>}
        {voices.isError && <div className="errline" style={{ margin: 14 }}>{voices.error.message}</div>}
        {voices.data?.unreadable.map((u) => (
          <div className="refusal" key={u.voice_id} style={{ margin: "10px 14px 0" }}>
            <span className="tag">unreadable</span>
            <p>{u.voice_id}: {u.error}</p>
          </div>
        ))}
        {voices.data && voices.data.voices.length === 0 && (
          <div className="loadline" style={{ padding: 14 }}>
            no voices yet — add a preset to get narrating, or clone from a reference clip
          </div>
        )}
        {voices.data && all.length > 0 && shown.length === 0 && (
          <div className="loadline" style={{ padding: 14 }}>
            nothing matches those filters — {all.length} voice(s) hidden
          </div>
        )}
        <div className="voices" style={{ padding: 14 }}>
          {shown.map((v) => <VoiceCardView key={v.voice_id} voice={v} titleFor={titleFor} />)}
        </div>
      </div>
      {dialog === "add" && <AddVoiceDialog onClose={() => setDialog("none")} />}
      {dialog === "clone" && <CloneDialog onClose={() => setDialog("none")} />}
    </section>
  );
}
