import { useState } from "react";

import { ApiError } from "../../api/client";
import { useCreateVoice, useKokoroPresets } from "../../api/hooks";
import type { VoiceCreate } from "../../api/types";
import { TalkDialog } from "../../components/Dialog";
import { TalkSelect } from "../../components/Select";
import { TalkSlider } from "../../components/Slider";
import type { DuplicateRecipe } from "./helpers";
import { mixPreviewUrl, presetPreviewUrl, useDemoPlayer } from "./useDemoPlayer";

/* -------------------------------------------------- add / clone dialogs */

export function AddVoiceDialog({ onClose, initial }: { onClose: () => void; initial?: DuplicateRecipe }) {
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
