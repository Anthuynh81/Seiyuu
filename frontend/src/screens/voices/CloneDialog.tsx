import { useRef, useState } from "react";

import { ApiError } from "../../api/client";
import { useCloneVoice } from "../../api/hooks";
import { TalkDialog } from "../../components/Dialog";
import { TalkSelect } from "../../components/Select";

export function CloneDialog({ onClose }: { onClose: () => void }) {
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
