import { Link } from "react-router-dom";

import { useBooks, useCharacterNames, useDeleteVoice, useVoiceReferences } from "../../api/hooks";
import type { VoiceOut, VoiceReferencesOut } from "../../api/types";
import { TalkDialog } from "../../components/Dialog";
import { JobErrorText } from "../../components/JobErrorText";
import { explainFailure, failureOf, SCREENS, type JobErrorExplanation } from "../../lib/glossary";
import { bookHref } from "../../lib/nextStep";

/* -------------------------------------------------- delete-voice confirmation */

const CHARACTER_PREFIX = "character:";

/** The server's refusals for this action, through the one glossary (lib/glossary), with the
    wording sharpened for this dialog: what did or didn't happen to the voice. The recourse
    (e.g. stop the job in the way) and the raw message under Details come from the glossary.
    `stage` says which call failed: loading the list happens before the user asked to delete
    anything, so its copy must not read as a failed delete. */
function explainDeleteError(err: unknown, stage: "check" | "delete"): JobErrorExplanation {
  const out = explainFailure(failureOf(err));
  switch (out.id) {
    case "voice_in_use":
      return {
        ...out,
        summary: "This voice is cast in a book, so it wasn't deleted.",
        fix: `The list above is up to date now. Give those parts another voice on ${SCREENS.review}, then try again.`,
      };
    case "render_active":
      return stage === "delete" ? { ...out, summary: `${out.summary} Nothing was deleted.` } : out;
    case "damaged_file":
      return {
        ...out,
        summary:
          stage === "check"
            ? "Seiyuu couldn't check which books use this voice, so it can't be deleted right now."
            : "Seiyuu couldn't check which books use this voice, so nothing was deleted.",
        fix: "One book's casting file can't be read (Details names it). Fix or remove that file, then try again.",
      };
    case "not_found":
      return { ...out, summary: "This voice was already deleted.", fix: null };
    default:
      return out;
  }
}

interface BookUse {
  bookId: string;
  title: string;
  roles: string[];
}

/** Group the reference rows by book, in plain words: titles instead of book ids, character
    names instead of character ids, narrator and inner-thoughts parts named as such. */
function groupUses(
  refs: VoiceReferencesOut["references"],
  titleOf: (bookId: string) => string,
  names: Map<string, Map<string, string>>,
): BookUse[] {
  const byBook = new Map<string, BookUse>();
  for (const ref of refs) {
    let use = byBook.get(ref.book_id);
    if (!use) {
      use = { bookId: ref.book_id, title: titleOf(ref.book_id), roles: [] };
      byBook.set(ref.book_id, use);
    }
    if (ref.role === "narrator") use.roles.push("Narrator");
    else if (ref.role === "thought") use.roles.push("Inner thoughts");
    else if (ref.role.startsWith(CHARACTER_PREFIX)) {
      const charId = ref.role.slice(CHARACTER_PREFIX.length);
      use.roles.push(names.get(ref.book_id)?.get(charId) ?? "A character");
    } else use.roles.push(ref.role);
  }
  return [...byBook.values()];
}

export function DeleteVoiceDialog({ voice, onClose }: { voice: VoiceOut; onClose: () => void }) {
  const refs = useVoiceReferences(voice.voice_id);
  const books = useBooks();
  const del = useDeleteVoice();

  const rows = refs.data?.references ?? [];
  const charBooks = [
    ...new Set(rows.filter((r) => r.role.startsWith(CHARACTER_PREFIX)).map((r) => r.book_id)),
  ];
  const names = useCharacterNames(charBooks);
  const titleOf = (bookId: string) => books.data?.books.find((b) => b.book_id === bookId)?.title ?? bookId;
  const uses = groupUses(rows, titleOf, names);

  const checked = refs.isSuccess && !refs.isFetching;
  const canDelete = checked && uses.length === 0 && !del.isPending;
  const cloned = voice.kind === "cloned";
  // while the DELETE is in flight the dialog stays put (no cancel, esc or click-outside):
  // closing it can't stop the request, and its result — success or refusal — shows only here
  const busy = del.isPending;

  return (
    <TalkDialog
      title="Delete voice"
      onClose={onClose}
      dismissable={!busy}
      footer={
        <>
          <button className="key quiet" disabled={busy} onClick={onClose}>
            cancel
          </button>
          <button
            className="key danger"
            disabled={!canDelete}
            onClick={() => del.mutate(voice.voice_id, { onSuccess: onClose })}
          >
            {del.isPending ? "deleting…" : "delete voice"}
          </button>
        </>
      }
    >
      <p className="m-0 text-ink-2">
        Delete <b className="text-ink">{voice.name}</b> from your voice library?
      </p>

      <div className="mono mb-1 mt-3.5 text-[10.5px] uppercase tracking-[0.08em] text-ink-3">Used by</div>
      {refs.isError ? (
        <JobErrorText explained={explainDeleteError(refs.error, "check")} />
      ) : !checked ? (
        <p className="m-0 text-ink-3">Checking which books use this voice…</p>
      ) : uses.length === 0 ? (
        <p className="m-0 text-ink-2">Not cast in any book.</p>
      ) : (
        <>
          <ul className="m-0 list-none p-0" aria-label="books using this voice">
            {uses.map((u) => (
              <li key={u.bookId} className="border-b border-hairline py-1 text-ink-2 last:border-b-0">
                <Link className="link" to={bookHref("review", u.bookId)} onClick={onClose}>
                  <b>{u.title}</b>
                </Link>{" "}
                — {u.roles.join(", ")}
              </li>
            ))}
          </ul>
          <p className="mb-0 mt-2 text-ink-2">
            A voice that's still cast can't be deleted. Give these parts another voice on {SCREENS.review}
            first — click a book to open it there.
          </p>
        </>
      )}

      {cloned ? (
        <p className="mb-0 mt-3.5 text-ink-2">
          <b className="text-clip">This can't be undone.</b> The reference clip this voice was cloned from — the
          recording you attested consent for — is deleted permanently. To use this voice again you would have to
          upload the clip and attest consent again.
        </p>
      ) : (
        <p className="mb-0 mt-3.5 text-ink-2">
          <b className="text-clip">This can't be undone.</b> The voice is removed from your library for good.
        </p>
      )}

      <p className="mb-0 mt-2 text-ink-3">
        Audiobooks and lines already recorded with this voice keep playing, but you can't record new or
        re-recorded lines with it once it's deleted.
      </p>

      {del.error && (
        <div className="mt-3">
          <JobErrorText explained={explainDeleteError(del.error, "delete")} />
        </div>
      )}
    </TalkDialog>
  );
}
