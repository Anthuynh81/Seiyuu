import { useRef, useState } from "react";

import { ApiError } from "../../api/client";
import { useBooks, useCloneVoice } from "../../api/hooks";
import { TalkDialog } from "../../components/Dialog";
import { TalkSelect } from "../../components/Select";
import { PlainError } from "../../components/PlainError";
import { explainFailure, failureOf, staleBooks } from "../../lib/glossary";
import { readStored, writeStored } from "../../lib/storage";

export function CloneDialog({ onClose }: { onClose: () => void }) {
  const clone = useCloneVoice();
  const fileInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [engine, setEngine] = useState("chatterbox");
  const [attested, setAttested] = useState(false);
  const [attestedBy, setAttestedBy] = useState(() => readStored("seiyuu.attestedBy") ?? "");
  const err = clone.error instanceof ApiError ? clone.error : null;
  const recloneBlocked = err?.code === "reclone_blocked";
  // the consent step names the blast radius BEFORE anything is replaced: the recorded books
  // that flip back to un-recorded (the backend's dry-run scan), by title
  const books = useBooks();
  const stale = recloneBlocked ? staleBooks(err!.detail) : [];
  const titleOf = (id: string) => books.data?.books.find((b) => b.book_id === id)?.title ?? id;
  const reclone = recloneBlocked ? explainFailure(failureOf(err)) : null;

  const submit = (replace = false) => {
    if (!file) return;
    writeStored("seiyuu.attestedBy", attestedBy);
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
      {clone.error && !recloneBlocked && <PlainError className="mt-3" error={clone.error} title="couldn't clone" />}
      {reclone && (
        <div className="refusal mt-3" role="alert">
          <span className="tag">already cloned</span>
          <p className="joberr-sum">{reclone.summary}</p>
          {stale.length > 0 ? (
            <>
              <p className="joberr-fix">
                {stale.length === 1 ? "This recorded book goes" : `These ${stale.length} recorded books go`} back to
                not recorded, and need re-recording:
              </p>
              <ul className="m-0 list-none p-0" aria-label="books that need re-recording">
                {stale.map((id) => (
                  <li key={id} className="border-b border-hairline py-1 text-ink-2 last:border-b-0">
                    <b>{titleOf(id)}</b>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="joberr-fix">No recorded book uses it yet.</p>
          )}
          <p className="joberr-fix">
            Its saved lines are cleared, and any paid lines are billed again when re-recorded.{" "}
            <button className="link" onClick={() => submit(true)}>
              replace it
            </button>
          </p>
        </div>
      )}
    </TalkDialog>
  );
}
