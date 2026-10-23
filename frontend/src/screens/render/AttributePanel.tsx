import { useEffect, useId, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { ApiError } from "../../api/client";
import {
  useAttribute,
  useCharacters,
  useSettings,
  useSystem,
  useSystemProbe,
  type AttributeInput,
} from "../../api/hooks";
import type { ActiveJobSummary, AttributionDefaults } from "../../api/types";
import { JobErrorText } from "../../components/JobErrorText";
import { TalkSelect } from "../../components/Select";
import {
  attributeBlocker,
  chapterSpan,
  parseChapterList,
  promptChanged,
  sameReaderReason,
  type AttributeProvider,
  type Blocker,
  type Reader,
} from "../../lib/attribute";
import { SCREENS } from "../../lib/glossary";
import { explainJobError, failureOf } from "../../lib/jobErrors";
import { bookHref } from "../../lib/nextStep";

/** "Full cast: find the speakers" — the attribute step, wherever the user meets it (a Library
    card, Review's empty state, the Render screen). The API is the only writer: this panel
    posts POST /books/{id}/attribute and reads every state back from the book's live job and
    the system probe. Paid runs (Claude) are never auto-confirmed: the server's 402 is shown
    with an explicit "confirm the paid run" control that re-sends confirm_paid. */

export interface AttributePanelBook {
  book_id: string;
  attributed: boolean;
  active_job: ActiveJobSummary | null;
}

const PROVIDER_LABEL: Record<AttributeProvider, string> = {
  local: "this PC (Ollama)",
  anthropic: "Claude online",
};

function defaultReader(defaults: AttributionDefaults | undefined): Reader | null {
  if (!defaults) return null;
  return { provider: defaults.provider === "anthropic" ? "anthropic" : "local", model: defaults.model };
}

/** Provider + model controls (the old Render "change" row, unchanged in what it sends). */
function ReaderPicker({
  reader,
  defaults,
  localModel,
  anthropicKey,
  onChange,
  label,
}: {
  reader: Reader;
  defaults: AttributionDefaults;
  /** The server's local model; `defaults.model` is the CLAUDE id when Claude is the default. */
  localModel: string | null;
  anthropicKey: boolean;
  onChange: (r: Reader) => void;
  label: string;
}) {
  return (
    <div className="attr-pick">
      <TalkSelect
        ariaLabel={`${label} provider`}
        value={reader.provider}
        onChange={(v) => {
          const p = v as AttributeProvider;
          // unknown local model (settings not loaded): blank, and the server uses its own default
          onChange({ provider: p, model: p === "local" ? (localModel ?? "") : defaults.anthropic_model });
        }}
        options={[
          { value: "local", label: "this PC (Ollama), free" },
          {
            value: "anthropic",
            label: `Claude online, paid${anthropicKey ? "" : " (no key set up)"}`,
            disabled: !anthropicKey,
          },
        ]}
      />
      <input
        type="text"
        value={reader.model}
        placeholder={reader.provider === "local" ? "default model" : undefined}
        aria-label={`${label} model`}
        onChange={(e) => onChange({ provider: reader.provider, model: e.target.value })}
        className="mono w-[180px] border border-hairline bg-booth-0 px-2 py-[5px] text-[11.5px] text-ink"
      />
    </div>
  );
}

/** A refusal from the enqueue, in plain words; a 402 gets the explicit paid confirm. */
function AttributeError({ error, onConfirmPaid }: { error: unknown; onConfirmPaid: () => void }) {
  if (!(error instanceof ApiError)) return null;
  if (error.code === "payment_confirmation_required") {
    // never auto-confirmed: the user clicks, and only then is confirm_paid sent
    return (
      <div className="mt-1.5">
        <p className="text-[12.5px] text-ink">
          This run uses Claude online, which costs money. Nothing has been spent yet.{" "}
          <button className="link" onClick={onConfirmPaid}>
            confirm the paid run
          </button>
        </p>
      </div>
    );
  }
  // every other refusal (Ollama pre-flight 503s, render_active, ...) reads in plain words,
  // with its recourse (e.g. Cancel recording) and the raw message under Details
  return (
    <div className="mt-1.5">
      <JobErrorText explained={explainJobError(failureOf(error))} />
    </div>
  );
}

function BlockerNote({ blocker, id }: { blocker: Blocker; id: string }) {
  return (
    <div className="attr-block" id={id}>
      <p>{blocker.reason}</p>
      <code className="chk-cmd">{blocker.command}</code>
    </div>
  );
}

const STARTED = "Started. Follow it in the bar at the bottom of the screen; you can leave this page.";

/** Announced (role=status) so a screen-reader user hears the start. */
function Started() {
  return (
    <p className="attr-ok" role="status">
      {STARTED}
    </p>
  );
}

export function AttributePanel({
  book,
  chapterCount,
  chapters = [],
  id,
}: {
  book: AttributePanelBook;
  /** The book's chapter count; the re-run's chapter list is checked against it. */
  chapterCount?: number;
  /** First-run scope ([] = the whole book). Render passes its scope row's chapters. */
  chapters?: number[];
  /** Anchor id: the next-step links land here (Render's #find-speakers). */
  id?: string;
}) {
  const headId = useId();
  const running = book.active_job?.kind === "attribute";
  const stripRef = useRef<HTMLDivElement>(null);
  const showedForm = useRef(false);
  // Starting swaps the form for the "finding the speakers" strip once the job shows up on the
  // book: the focused start button leaves the DOM. Hand focus to the strip so the keyboard
  // user isn't dropped to the top of the page. Only after the form was shown here (never
  // steal focus on a page that loads with the job already running).
  useEffect(() => {
    if (!running) {
      showedForm.current = true;
      return;
    }
    const lost = document.activeElement === null || document.activeElement === document.body;
    if (showedForm.current && lost) stripRef.current?.focus();
  }, [running]);

  if (running && book.active_job) {
    return (
      <div className="drainstrip mb-3.5" id={id} tabIndex={-1} ref={stripRef} role="status">
        <span className="state"><i className={`led ${book.active_job.state === "queued" ? "q" : "run"}`} />finding the speakers</span>
        <span className="mono text-[11.5px] text-ink-2">
          {book.active_job.state === "queued"
            ? "waiting for the job ahead of it — it starts on its own"
            : `follow it in the bar at the bottom — ${SCREENS.review} and the read-along unlock when it lands`}
        </span>
      </div>
    );
  }

  return (
    <section className="panel attrpanel" id={id} tabIndex={-1} aria-labelledby={headId}>
      {book.attributed ? (
        <RerunBody book={book} chapterCount={chapterCount} headId={headId} />
      ) : (
        <FirstRunBody book={book} chapters={chapters} headId={headId} />
      )}
    </section>
  );
}

/** Shared state for both bodies: the server defaults, the live probe, the enqueue. */
function useAttributeRun(bookId: string) {
  const system = useSystem();
  const probe = useSystemProbe();
  const attribute = useAttribute(bookId);
  const defaults = system.data?.attribution;
  // with Claude as the default, /api/system names only the Claude model: the local one (what
  // the probe checks, what "this PC" should read with) comes from /api/settings
  const settings = useSettings({ enabled: !!defaults && defaults.provider !== "local" });
  const localModel = !defaults
    ? null
    : defaults.provider === "local"
      ? defaults.model
      : (settings.data?.attribution_model ?? null);
  return { system, probe, attribute, localModel };
}

function FirstRunBody({ book, chapters, headId }: { book: AttributePanelBook; chapters: number[]; headId: string }) {
  const { system, probe, attribute, localModel } = useAttributeRun(book.book_id);
  const blockId = useId();
  const [llm, setLlm] = useState<Reader | null>(null);
  const [llmOpen, setLlmOpen] = useState(false);
  const attrDefaults = system.data?.attribution;
  const effLlm = llm ?? defaultReader(attrDefaults);
  const blocker = attributeBlocker({ probe: probe.data, probeError: probe.error, reader: effLlm, localModel });

  const start = (confirmPaid = false) =>
    attribute.mutate({
      chapters,
      ...(effLlm ? { provider: effLlm.provider } : {}),
      // a blank model means "the server's default for that provider"
      ...(effLlm?.model.trim() ? { model: effLlm.model.trim() } : {}),
      ...(confirmPaid ? { confirm_paid: true } : {}),
    });
  const scope = chapters.length ? `ch ${chapters[0]}–${chapters[chapters.length - 1]}` : "the whole book";

  return (
    <>
      <div className="panel-h">
        <b id={headId}>Full cast: find the speakers</b>
        {/* hybrid escalates hard blocks to Claude: the server asks for approval either way */}
        <span className="tag ml-auto">{effLlm?.provider === "anthropic" || attrDefaults?.hybrid ? "paid" : "free"}</span>
      </div>
      <div className="attr-body">
        <p className="attr-why">
          Seiyuu reads the book to work out who says each line, so every character can get a voice. To record it all in
          one voice, you can skip this.
        </p>
        <button
          className="key"
          disabled={attribute.isPending || attribute.isSuccess || blocker !== null}
          aria-describedby={blocker ? blockId : undefined}
          onClick={() => start()}
        >
          {attribute.isPending ? "starting…" : `▶ find the speakers in ${scope}`}
        </button>
        {blocker && <BlockerNote blocker={blocker} id={blockId} />}
        <p className="attr-reader">
          reads with:{" "}
          {effLlm
            ? `${PROVIDER_LABEL[effLlm.provider]} · ${effLlm.model.trim() || "default model"}`
            : "the default from your settings"}{" "}
          {effLlm && attrDefaults && (
            <button className="link" aria-expanded={llmOpen} onClick={() => setLlmOpen(!llmOpen)}>
              {llmOpen ? "done" : "change"}
            </button>
          )}
        </p>
        {llmOpen && effLlm && attrDefaults && (
          <ReaderPicker
            label="speaker-finding"
            reader={effLlm}
            defaults={attrDefaults}
            localModel={localModel}
            anthropicKey={!!system.data?.keys.anthropic_configured}
            onChange={(r) => {
              setLlm(r);
              attribute.reset(); // a refusal (or a paid confirm) was about the old choice
            }}
          />
        )}
        <AttributeError error={attribute.error} onConfirmPaid={() => start(true)} />
        {attribute.isSuccess && <Started />}
      </div>
    </>
  );
}

function RerunBody({
  book,
  chapterCount,
  headId,
}: {
  book: AttributePanelBook;
  chapterCount: number | undefined;
  headId: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className="panel-h">
        <i className="led ok" aria-hidden />
        <b id={headId}>Speakers found</b>
        <Link className="link ml-auto text-xs" to={bookHref("review", book.book_id)}>
          check them on {SCREENS.review}
        </Link>
      </div>
      <details className="fine attr-adv" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary>Advanced: re-run speaker finding for chapters…</summary>
        {open && <RerunForm book={book} chapterCount={chapterCount ?? 0} />}
      </details>
    </>
  );
}

function RerunForm({ book, chapterCount }: { book: AttributePanelBook; chapterCount: number }) {
  const { system, probe, attribute, localModel } = useAttributeRun(book.book_id);
  // the reader that produced the current speakers (the report's provenance) — fetched only
  // once this disclosure is open
  const overview = useCharacters(book.book_id, true);
  const reasonId = useId();
  const [text, setText] = useState("");
  const [picked, setPicked] = useState<Reader | null>(null);
  const attrDefaults = system.data?.attribution;
  const reader = picked ?? defaultReader(attrDefaults);
  const current = overview.data
    ? {
        provider_id: overview.data.provider_id,
        model_id: overview.data.model_id,
        prompt_version: overview.data.prompt_version,
      }
    : null;
  const promptNow = attrDefaults?.prompt_version;

  const parsed = parseChapterList(text, chapterCount);
  const sameReason = reader ? sameReaderReason(current, reader, promptNow) : null;
  const blocker = attributeBlocker({ probe: probe.data, probeError: probe.error, reader, localModel });
  // ONE reason at a time, in the order the user fixes them. While the overview refetches
  // (e.g. right after a re-run finished) its provenance may still be the previous run's: wait.
  const reason = !reader
    ? "Loading the reader settings…"
    : !overview.data || overview.isFetching
      ? overview.isError && !overview.isFetching
        ? "Couldn't load which model found these speakers. Try again in a moment."
        : "Checking which model found these speakers…"
      : text.trim() === ""
        ? "List the chapters to re-read, for example 3, 5-7."
        : (parsed.error ?? sameReason ?? null);

  const run = (confirmPaid = false) => {
    if (!reader || !parsed.chapters) return;
    const input: AttributeInput = {
      chapters: parsed.chapters,
      provider: reader.provider,
      model: reader.model.trim(),
      ...(confirmPaid ? { confirm_paid: true } : {}),
    };
    attribute.mutate(input);
  };
  const disabled = attribute.isPending || attribute.isSuccess || reason !== null || blocker !== null;

  return (
    <div className="attr-body">
      <ul className="attr-notes">
        <li>Only the chapters you list are read again; the rest of the book keeps its speakers. Your speaker fixes are re-applied afterwards.</li>
        <li>Seiyuu remembers earlier answers, so the same model gives the same result. Choose a different model or provider.</li>
        <li>Merging two names for the same character only runs when the whole book is read, so it is skipped here.</li>
      </ul>
      <p className="attr-reader">
        found with: {current ? `${PROVIDER_LABEL[current.provider_id as AttributeProvider] ?? current.provider_id} · ${current.model_id}` : "…"}
      </p>
      {promptChanged(current, promptNow) && (
        <p className="attr-why">
          Seiyuu's speaker-finding instructions have been updated since these were found, so even the same model
          reads these chapters afresh.
        </p>
      )}
      <label className="attr-field">
        <span className="tag">chapters</span>
        <input
          type="text"
          value={text}
          placeholder="e.g. 3, 5-7"
          aria-label="chapters to re-read"
          onChange={(e) => {
            setText(e.target.value);
            attribute.reset();
          }}
          className="mono w-[160px] border border-hairline bg-booth-0 px-2 py-[5px] text-[11.5px] text-ink"
        />
        <span className="mono text-[11px] text-ink-3">of {chapterCount}</span>
      </label>
      {reader && attrDefaults && (
        <ReaderPicker
          label="re-run"
          reader={reader}
          defaults={attrDefaults}
          localModel={localModel}
          anthropicKey={!!system.data?.keys.anthropic_configured}
          onChange={(r) => {
            setPicked(r);
            attribute.reset();
          }}
        />
      )}
      <button
        className="key mt-2"
        disabled={disabled}
        aria-describedby={reason || blocker ? reasonId : undefined}
        onClick={() => run()}
      >
        {attribute.isPending
          ? "starting…"
          : `re-run for ${parsed.chapters ? chapterSpan(parsed.chapters) : "these chapters"}`}
      </button>
      {reason ? (
        <p className="attr-why mt-1.5" id={reasonId}>
          {reason}
        </p>
      ) : (
        blocker && <BlockerNote blocker={blocker} id={reasonId} />
      )}
      <AttributeError error={attribute.error} onConfirmPaid={() => run(true)} />
      {attribute.isSuccess && <Started />}
    </div>
  );
}
