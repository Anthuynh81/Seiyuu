import { useRef, useState } from "react";

import { ApiError } from "../api/client";
import {
  useAudition,
  useCloneVoice,
  useCloudSlots,
  useCreateVoice,
  useDeleteVoice,
  useKokoroPresets,
  useVoices,
  useWarmup,
} from "../api/hooks";
import type { VoiceOut } from "../api/types";

/* -------------------------------------------------- audition control */

function AuditionControl({ voice }: { voice: VoiceOut }) {
  const audition = useAudition(voice.voice_id);
  const warmup = useWarmup();
  const [playerOpen, setPlayerOpen] = useState(false);
  const err = audition.error instanceof ApiError ? audition.error : null;

  if (audition.isPending) {
    return (
      <div className="audit">
        <span className="stop" />
        <span className="lbl">auditioning…</span>
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
              style={{ background: "none", border: "none", color: "var(--tungsten)", cursor: "pointer", padding: 0 }}
              disabled={warmup.isPending}
              onClick={() => warmup.mutate(voice.engine, { onSuccess: () => audition.reset() })}
            >
              {warmup.isPending ? "starting warmup…" : "warm up first"}
            </button>
          );
        case "payment_confirmation_required":
          return (
            <button
              className="link"
              style={{ background: "none", border: "none", color: "var(--tungsten)", cursor: "pointer", padding: 0 }}
              onClick={() => audition.mutate(true)}
            >
              confirm ~${Number(detail.estimated_usd ?? 0).toFixed(4)} &amp; play
            </button>
          );
        case "gpu_busy":
        case "cloud_busy":
          return <span>wait for the job in the transport bar, or cancel it — <button className="link" style={{ background: "none", border: "none", color: "var(--tungsten)", cursor: "pointer", padding: 0 }} onClick={() => audition.reset()}>retry</button></span>;
        default:
          return (
            <button className="link" style={{ background: "none", border: "none", color: "var(--tungsten)", cursor: "pointer", padding: 0 }} onClick={() => audition.reset()}>
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
      <div className="audit" role="button" tabIndex={0} onClick={() => audition.mutate(false)} style={{ cursor: "pointer" }}>
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

/* -------------------------------------------------- voice card */

function VoiceCardView({ voice }: { voice: VoiceOut }) {
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
  const error = create.error instanceof ApiError ? create.error.message : create.error?.message;

  const submit = () => {
    const body =
      kind === "preset"
        ? ({ kind, name, engine, preset_id: engine === "kokoro" ? presetId : cloudId } as const)
        : ({ kind, name, gender: gender || null, accent } as const);
    create.mutate(body, { onSuccess: onClose });
  };

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
            <button className={`chap ${kind === "blend" ? "on" : ""}`} onClick={() => setKind("blend")}>blend (auto recipe)</button>
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
                  <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
                    {(presets.data?.voices ?? [{ id: "af_heart", name: "Heart", language: "en-US", gender: "female" }]).map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.id} — {p.gender ?? "?"} {p.language ?? ""}
                      </option>
                    ))}
                  </select>
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
          )}
          {error && <div className="errline" style={{ marginTop: 12 }}>{error}</div>}
        </div>
        <div className="df">
          <button className="key quiet" onClick={onClose}>cancel</button>
          <button className="key" disabled={create.isPending || !name.trim() || (kind === "preset" && engine === "elevenlabs" && !cloudId.trim())} onClick={submit}>
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

export function Voices() {
  const voices = useVoices();
  const slots = useCloudSlots();
  const [dialog, setDialog] = useState<"none" | "add" | "clone">("none");

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
        <div className="voices" style={{ padding: 14 }}>
          {voices.data?.voices.map((v) => <VoiceCardView key={v.voice_id} voice={v} />)}
        </div>
      </div>
      {dialog === "add" && <AddVoiceDialog onClose={() => setDialog("none")} />}
      {dialog === "clone" && <CloneDialog onClose={() => setDialog("none")} />}
    </section>
  );
}
