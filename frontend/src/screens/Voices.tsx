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
  useRenameVoice,
  useSetVoiceTags,
  useVoices,
  useWarmup,
} from "../api/hooks";
import type { VoiceCreate, VoiceOut } from "../api/types";
import { TalkDialog } from "../components/Dialog";
import { TalkSelect } from "../components/Select";
import { TalkSlider } from "../components/Slider";

/* -------------------------------------------------- audition control */

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
              disabled={warmup.isPending}
              onClick={() => warmup.mutate(voice.engine, { onSuccess: () => audition.reset() })}
            >
              {warmup.isPending ? "starting warmup…" : "warm up first"}
            </button>
          );
        case "payment_confirmation_required":
          return (
            <button className="link" onClick={() => startAudition(true)}>
              confirm ~${Number(detail.estimated_usd ?? 0).toFixed(4)} &amp; play
            </button>
          );
        case "gpu_busy_retry":
          // auto-retries exhausted — the render is still holding the GPU; let the user retry
          return (
            <span>
              the render keeps running — it hasn't yielded the GPU yet;{" "}
              <button className="link" onClick={() => startAudition(audition.variables ?? false)}>
                retry
              </button>
            </span>
          );
        case "gpu_busy":
        case "cloud_busy":
          return <span>wait for the job in the transport bar, or cancel it — <button className="link" onClick={() => audition.reset()}>retry</button></span>;
        default:
          return (
            <button className="link" onClick={() => audition.reset()}>
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
      <div className="audit cursor-pointer" role="button" tabIndex={0} onClick={() => startAudition(false)}>
        <span className="play" />
        <span className="lbl">audition</span>
        {voice.has_audition && (
          <button
            className="key quiet ml-auto px-2 py-px text-[10.5px]"
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
          className="mt-2 h-[30px] w-full"
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
          className="rowedit visible ml-0"
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
      <button className="key quiet px-2 py-[2px]" disabled={setTags.isPending} onClick={save}>
        save
      </button>
      {setTags.error && <span className="mono text-[10px] text-clip">{setTags.error.message}</span>}
    </div>
  );
}

/* -------------------------------------------------- rename (name is a safe, cache-free label) */

function NameRow({ voice }: { voice: VoiceOut }) {
  const rename = useRenameVoice();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(voice.name);
  const save = () => {
    const name = draft.trim();
    if (!name || name === voice.name) return setEditing(false);
    rename.mutate({ voiceId: voice.voice_id, name }, { onSuccess: () => setEditing(false) });
  };
  if (!editing) {
    return (
      <h3 className="namerow">
        {voice.name}
        <button
          className="rowedit visible ml-1.5"
          title="rename voice"
          aria-label="rename voice"
          onClick={() => {
            setDraft(voice.name);
            rename.reset();
            setEditing(true);
          }}
        >
          ✎
        </button>
      </h3>
    );
  }
  return (
    <h3 className="namerow">
      <input
        className="taginput flex-1"
        value={draft}
        autoFocus
        aria-label="voice name"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") setEditing(false);
        }}
      />
      <button className="key quiet px-2 py-[2px]" disabled={rename.isPending} onClick={save}>
        save
      </button>
      {rename.error && <span className="mono text-[10px] text-clip">{rename.error.message}</span>}
    </h3>
  );
}

/* -------------------------------------------------- voice card */

/** A recipe lifted off an existing preset/blend voice to pre-fill the Add dialog. Duplicating
    mints a NEW voice_id with its own (empty) render history — so tweaking the copy can never
    drift the original's cached audio. Cloned voices can't be duplicated this way (their source
    is reference.wav + a hash-bound consent, which the create path can't re-derive). */
export interface DuplicateRecipe {
  kind: "preset" | "blend";
  name: string;
  engine: string;
  presetId?: string;
  layers?: { preset_id: string; weight: number }[];
  seed: number;
}

function recipeOf(voice: VoiceOut): DuplicateRecipe | null {
  if (voice.kind === "preset" && voice.preset_id) {
    return { kind: "preset", name: `${voice.name} copy`, engine: voice.engine, presetId: voice.preset_id, seed: voice.seed };
  }
  if (voice.kind === "blend" && voice.blend && voice.blend.length >= 2) {
    // weights are stored as normalized ratios; scale to readable 0-100 for the mixer faders
    const sum = voice.blend.reduce((a, b) => a + b.weight, 0) || 1;
    const layers = voice.blend.map((b) => ({ preset_id: b.preset_id, weight: Math.max(1, Math.round((100 * b.weight) / sum)) }));
    return { kind: "blend", name: `${voice.name} copy`, engine: voice.engine, layers, seed: voice.seed };
  }
  return null;
}

function VoiceCardView({
  voice,
  titleFor,
  onDuplicate,
}: {
  voice: VoiceOut;
  titleFor: (tag: string) => string;
  onDuplicate: (recipe: DuplicateRecipe) => void;
}) {
  const del = useDeleteVoice();
  const recipe = recipeOf(voice);
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
          className="rowedit visible float-right"
          title="delete voice"
          disabled={del.isPending}
          onClick={() => del.mutate(voice.voice_id)}
        >
          ✕
        </button>
        {recipe && (
          <button
            className="rowedit visible float-right mr-1.5"
            title="duplicate — opens Add voice pre-filled with this recipe (a new, freely-editable voice)"
            aria-label="duplicate voice"
            onClick={() => onDuplicate(recipe)}
          >
            ⧉
          </button>
        )}
      </span>
      <NameRow voice={voice} />
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
        <span className="consent invisible">.</span>
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

function AddVoiceDialog({ onClose, initial }: { onClose: () => void; initial?: DuplicateRecipe }) {
  const create = useCreateVoice();
  const presets = useKokoroPresets();
  const [kind, setKind] = useState<"preset" | "blend">(initial?.kind ?? "preset");
  const [name, setName] = useState(initial?.name ?? "");
  const [engine, setEngine] = useState(initial?.engine ?? "kokoro");
  const [presetId, setPresetId] = useState(initial?.presetId ?? "af_heart");
  const [cloudId, setCloudId] = useState(initial?.kind === "preset" && initial.engine !== "kokoro" ? (initial.presetId ?? "") : "");
  const [gender, setGender] = useState("");
  const [accent, setAccent] = useState<"a" | "b">("a");
  // manual mix: layers of (preset, weight) — the server normalizes weights, so only
  // the ratios matter; the mixer shows the resulting percentages live. A duplicated blend
  // lands in manual mode with the source's resolved layers so it can be tweaked directly.
  const [manual, setManual] = useState(!!initial?.layers);
  const [layers, setLayers] = useState<{ preset_id: string; weight: number }[]>(
    initial?.layers ?? [
      { preset_id: "af_heart", weight: 60 },
      { preset_id: "af_nicole", weight: 40 },
    ],
  );
  const error = create.error instanceof ApiError ? create.error.message : create.error?.message;

  const demo = useDemoPlayer();
  const catalog =
    presets.data?.voices ??
    [{ id: "af_heart", name: "Heart", language: "en-US", gender: "female", description: null }];
  const describe = (id: string) => catalog.find((p) => p.id === id)?.description;
  const presetOptions = catalog.map((p) => ({
    value: p.id,
    label: `${p.id} — ${p.gender ?? "?"} ${p.language ?? ""}${p.description ? ` · ${p.description}` : ""}`,
  }));
  const demoKey = (url: string, label = "▶") => (
    <button
      className="key quiet px-[9px] py-[3px]"
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

  // a duplicate carries the source seed so the copy starts identical-sounding before tuning
  const seedPart = initial ? { seed: initial.seed } : {};
  const submit = () => {
    const body: VoiceCreate =
      kind === "preset"
        ? { kind, name, engine, preset_id: engine === "kokoro" ? presetId : cloudId, ...seedPart }
        : manual
          ? { kind, name, components: layers.filter((l) => l.weight > 0), ...seedPart }
          : { kind, name, gender: gender || null, accent, ...seedPart };
    create.mutate(body, { onSuccess: onClose });
  };

  const active = layers.filter((l) => l.weight > 0);
  // kokoro can't blend across language families (the id's first letter: af_/am_ = American,
  // bf_/bm_ = British) — catch it at the fader instead of a server refusal
  const familyMix = new Set(active.map((l) => l.preset_id[0])).size > 1;
  const blendInvalid = kind === "blend" && manual && (active.length < 2 || familyMix);

  return (
    <TalkDialog
      title={initial ? "Duplicate voice" : "Add voice"}
      onClose={onClose}
      footer={
        <>
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
            {initial ? "create copy" : "add voice"}
          </button>
        </>
      }
    >
      {initial && (
        <div className="tag mb-2 normal-case tracking-[0.02em] text-ink-3">
          creates a new voice from this recipe — the original stays exactly as it is
        </div>
      )}
      <div className="modewrap mb-1">
        <button className={`chap ${kind === "preset" ? "on" : ""}`} onClick={() => setKind("preset")}>preset</button>
        <button className={`chap ${kind === "blend" ? "on" : ""}`} onClick={() => setKind("blend")}>blend</button>
      </div>
      <label>voice name</label>
      <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Narrator" />
      {kind === "preset" ? (
        <>
          <label>engine</label>
          <TalkSelect
            ariaLabel="engine"
            value={engine}
            onChange={setEngine}
            options={[
              { value: "kokoro", label: "kokoro — local, free" },
              { value: "elevenlabs", label: "elevenlabs — cloud stock voice, paid to render" },
            ]}
          />
          {engine === "kokoro" ? (
            <>
              <label>preset</label>
              <div className="flex items-center gap-2">
                <TalkSelect className="flex-1" ariaLabel="preset" value={presetId} onChange={setPresetId} options={presetOptions} />
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
          <div className="modewrap mb-1.5">
            <button className={`chap ${!manual ? "on" : ""}`} onClick={() => setManual(false)}>auto — from name</button>
            <button className={`chap ${manual ? "on" : ""}`} onClick={() => setManual(true)}>manual mix</button>
          </div>
          {!manual ? (
            <>
              <label>gender hint for the recipe (optional)</label>
              <TalkSelect
                ariaLabel="gender hint"
                value={gender || "unknown"}
                onChange={(v) => setGender(v === "unknown" ? "" : v)}
                options={[
                  { value: "unknown", label: "unknown" },
                  { value: "female", label: "female" },
                  { value: "male", label: "male" },
                ]}
              />
              <label>accent</label>
              <TalkSelect
                ariaLabel="accent"
                value={accent}
                onChange={(v) => setAccent(v as "a" | "b")}
                options={[
                  { value: "a", label: "American" },
                  { value: "b", label: "British" },
                ]}
              />
            </>
          ) : (
            <>
              {layers.map((l, i) => (
                <div key={i}>
                  <div className="mixrow">
                    {demoKey(presetPreviewUrl(l.preset_id))}
                    <TalkSelect
                      className="w-[190px]"
                      ariaLabel={`layer ${i + 1} preset`}
                      value={l.preset_id}
                      onChange={(v) => setLayer(i, { preset_id: v })}
                      options={presetOptions}
                    />
                    <TalkSlider
                      className="flex-1"
                      value={l.weight}
                      onChange={(w) => setLayer(i, { weight: w })}
                      ariaLabel={`weight of ${l.preset_id}`}
                    />
                    <span className="pct">{Math.round((100 * l.weight) / totalWeight)}%</span>
                    <button
                      className="rowedit visible"
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
              <div className="mt-1.5 flex items-center gap-2.5">
                <button
                  className="key quiet px-[9px] py-[3px]"
                  onClick={() => setLayers([...layers, { preset_id: catalog[0].id, weight: 30 }])}
                >
                  + add layer
                </button>
                {!blendInvalid && demoKey(mixPreviewUrl(active), "▶ demo mix")}
                <span className="mono text-[10.5px] text-ink-3">
                  faders are ratios — the mix normalizes itself
                </span>
              </div>
              {blendInvalid && (
                <div className="mono mt-1.5 text-[11px] text-caution">
                  {familyMix
                    ? "kokoro can't blend across accents — keep every layer American (a…) or every layer British (b…)"
                    : "a blend needs at least two layers with weight"}
                </div>
              )}
            </>
          )}
        </>
      )}
      {demo.error && <div className="errline mt-3">{demo.error}</div>}
      {error && <div className="errline mt-3">{error}</div>}
    </TalkDialog>
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
    <TalkDialog
      title="New cloned voice"
      onClose={onClose}
      footer={
        <>
          <button className="key quiet" onClick={onClose}>cancel</button>
          <button className="key" disabled={clone.isPending || !file || !name.trim() || !attested || !attestedBy.trim()} onClick={() => submit(false)}>
            {clone.isPending ? "cloning…" : "clone voice"}
          </button>
        </>
      }
    >
      <label>reference clip (wav/mp3, ≥ 20 s recommended)</label>
      <div className="flex gap-2">
        <input type="text" readOnly value={file?.name ?? ""} placeholder="choose a file…" onClick={() => fileInput.current?.click()} />
        <button className="key quiet" onClick={() => fileInput.current?.click()}>browse</button>
        <input ref={fileInput} type="file" accept="audio/*" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>
      <label>voice name</label>
      <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Mr. Darcy" />
      <label>engine</label>
      <TalkSelect
        ariaLabel="engine"
        value={engine}
        onChange={setEngine}
        options={[
          { value: "chatterbox", label: "chatterbox — local, free" },
          { value: "indextts2", label: "indextts2 — local, emotion + cloning (slow, high quality)" },
          { value: "elevenlabs", label: "elevenlabs — cloud IVC, paid" },
        ]}
      />
      <div className="paper release slim">
        <div className="att">
          <input type="checkbox" id="att" checked={attested} onChange={(e) => setAttested(e.target.checked)} />
          <label htmlFor="att" className="attlbl m-0">
            I have the speaker's permission to clone this voice
          </label>
          <span className="mono ml-auto text-[11px] text-paper-ink-2">
            as{" "}
            <input
              type="text"
              value={attestedBy}
              onChange={(e) => setAttestedBy(e.target.value)}
              placeholder="your name"
              className="attname"
            />
          </span>
        </div>
      </div>
      <div className="tag mt-2 normal-case tracking-[0.02em] text-ink-3">
        one click — the attestation binds to these exact bytes (sha-256) and is required by the render gate
      </div>
      {err && !recloneBlocked && <div className="errline mt-3">{err.message}</div>}
      {recloneBlocked && (
        <div className="refusal mt-3">
          <span className="tag">reclone_blocked</span>
          <p>
            {err!.message} —{" "}
            <button className="link" onClick={() => submit(true)}>
              replace it (purges its cached audio; paid segments re-bill)
            </button>
          </p>
        </div>
      )}
    </TalkDialog>
  );
}

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
