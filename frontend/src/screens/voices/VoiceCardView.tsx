import { ApiError } from "../../api/client";
import { useDeleteVoice } from "../../api/hooks";
import type { VoiceOut } from "../../api/types";
import { AuditionControl } from "./AuditionControl";
import { type DuplicateRecipe, recipeOf } from "./helpers";
import { NameRow } from "./NameRow";
import { TagEditor } from "./TagEditor";

/* -------------------------------------------------- voice card */

export function VoiceCardView({
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
