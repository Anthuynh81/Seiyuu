import { useState } from "react";

import { useDesign, useJob, useKeepDesignTake, useStartDesign } from "../../api/hooks";
import type { DesignOut, DesignTakeOut, VoiceOut } from "../../api/types";
import { TalkDialog } from "../../components/Dialog";
import { JobErrorText } from "../../components/JobErrorText";
import { PlainError } from "../../components/PlainError";
import { TalkSelect } from "../../components/Select";
import { explainJobError } from "../../lib/jobErrors";
import { takeVerdict } from "./helpers";

/** Describe a voice → Qwen3-TTS VoiceDesign makes a few takes (a GPU job) → listen → keep one.
    The kept take becomes a library voice that every later line is cloned from. */
export function DesignDialog({ onClose }: { onClose: () => void }) {
  const start = useStartDesign();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [sample, setSample] = useState("");
  const [takes, setTakes] = useState("3");
  const [started, setStarted] = useState<{ designId: string; jobId: string } | null>(null);

  const job = useJob(started?.jobId ?? null);
  const live = started !== null && !job.data?.is_terminal;
  const design = useDesign(started?.designId ?? null, live);

  const submit = () =>
    start.mutate(
      {
        name: name.trim(),
        description: description.trim(),
        sample_text: sample.trim() || undefined,
        takes: Number(takes),
      },
      { onSuccess: (out) => setStarted({ designId: out.design.design_id, jobId: out.job.job_id }) },
    );

  if (started) {
    return (
      <TalkDialog
        title="Describe a voice"
        onClose={onClose}
        footer={<button className="key" onClick={onClose}>{live ? "close (keeps designing)" : "done"}</button>}
      >
        <Takes
          design={design.data}
          live={live}
          progress={job.data?.progress_text ?? ""}
          failed={job.data?.state === "failed" ? (job.data.error ?? "the design job failed") : null}
          stopped={job.data?.state === "canceled"}
        />
      </TalkDialog>
    );
  }

  const ready = name.trim() !== "" && description.trim().length >= 3 && !start.isPending;
  return (
    <TalkDialog
      title="Describe a voice"
      onClose={onClose}
      footer={
        <>
          <button className="key quiet" onClick={onClose}>cancel</button>
          <button className="key" disabled={!ready} onClick={submit}>
            {start.isPending ? "starting…" : "design it"}
          </button>
        </>
      }
    >
      <label htmlFor="design-name">voice name</label>
      <input
        id="design-name"
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Old janitor"
      />
      <label htmlFor="design-description">describe the voice</label>
      <textarea
        id="design-description"
        rows={3}
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="A gruff man in his sixties, slow and tired, a little hoarse"
      />
      <label htmlFor="design-sample">a line for it to say (optional)</label>
      <textarea
        id="design-sample"
        rows={2}
        value={sample}
        onChange={(e) => setSample(e.target.value)}
        placeholder="leave blank for a neutral line; a line in the character's style helps"
      />
      <label>takes to choose from</label>
      <TalkSelect
        ariaLabel="takes"
        value={takes}
        onChange={setTakes}
        options={[1, 2, 3, 4, 5].map((n) => ({ value: String(n), label: `${n} take${n === 1 ? "" : "s"}` }))}
      />
      <div className="tag mt-2 normal-case tracking-[0.02em] text-ink-3">
        uses the graphics card: about a minute to load the voice model, then a few seconds per take
      </div>
      {start.error && <PlainError className="mt-3" error={start.error} title="couldn't start the design" />}
    </TalkDialog>
  );
}

function Takes({
  design,
  live,
  progress,
  failed,
  stopped,
}: {
  design: DesignOut | undefined;
  live: boolean;
  progress: string;
  failed: string | null;
  stopped: boolean;
}) {
  const [kept, setKept] = useState<VoiceOut | null>(null);
  if (!design) return <p className="m-0 text-ink-3">Starting…</p>;
  return (
    <>
      <p className="m-0 text-ink-2">
        <b>{design.name}</b>: {design.description}
      </p>
      <p className="mono mt-1 text-[12px] text-ink-3">“{design.text}”</p>
      {live && (
        <p role="status" className="mt-2 text-ink-2">
          {progress || "waiting for the graphics card…"} ({design.takes.length} of {design.takes_requested} ready)
        </p>
      )}
      {failed && (
        <div className="refusal mt-2" role="alert">
          <span className="tag">design stopped</span>
          <JobErrorText explained={explainJobError(failed)} />
          {design.takes.length > 0 && <p className="joberr-fix">The takes it finished are still here.</p>}
        </div>
      )}
      {stopped && <p className="mt-2 text-ink-3">Stopped. The takes it finished are below.</p>}
      <ul className="m-0 mt-2 list-none p-0" aria-label="takes">
        {design.takes.map((t) => (
          <TakeRow key={t.take} designId={design.design_id} take={t} name={design.name} recommended={t.take === design.recommended} onKept={setKept} />
        ))}
      </ul>
      {kept && (
        <p role="status" className="mt-2 text-ink-2">
          Saved <b>{kept.name}</b> to your voices. Cast it on any character.
        </p>
      )}
    </>
  );
}

function TakeRow({
  designId,
  take,
  name,
  recommended,
  onKept,
}: {
  designId: string;
  take: DesignTakeOut;
  name: string;
  recommended: boolean;
  onKept: (voice: VoiceOut) => void;
}) {
  const keep = useKeepDesignTake(designId);
  return (
    <li className="border-b border-hairline py-2 last:border-b-0" aria-label={`take ${take.take}`}>
      <div className="flex items-center gap-2">
        <b>take {take.take}</b>
        {recommended && <span className="tag">recommended</span>}
        <span className="text-[12px] text-ink-3">{takeVerdict(take)}</span>
        {take.kept_voice_id ? (
          <span className="ml-auto text-[12px] text-ink-2">kept</span>
        ) : (
          <button
            className="key ml-auto"
            disabled={keep.isPending}
            onClick={() => keep.mutate({ take: take.take, name }, { onSuccess: onKept })}
          >
            {keep.isPending ? "saving…" : "keep this one"}
          </button>
        )}
      </div>
      <audio controls preload="none" src={take.audio_url} className="mt-1 h-[30px] w-full" />
      {keep.error && <PlainError className="mt-1" error={keep.error} title="couldn't keep it" />}
    </li>
  );
}
