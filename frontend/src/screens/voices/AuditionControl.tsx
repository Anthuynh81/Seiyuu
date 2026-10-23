import { useState } from "react";

import type { VoiceOut } from "../../api/types";
import { AuditionNotice } from "./AuditionNotice";
import { useAuditionFlow } from "./useAuditionFlow";

/* -------------------------------------------------- audition control */

export function AuditionControl({ voice }: { voice: VoiceOut }) {
  const flow = useAuditionFlow({ voiceId: voice.voice_id, engine: voice.engine });
  const [playerOpen, setPlayerOpen] = useState(false);

  if (flow.phase === "making") {
    return (
      <div className="audit">
        <span className="stop" />
        <span className="lbl">auditioning…</span>
        <span className="live" />
      </div>
    );
  }

  // waiting (soft borrow-retry) and refusals share their words and recourses with every player
  if (flow.phase === "waiting" || flow.phase === "refused") return <AuditionNotice flow={flow} />;

  return (
    <div>
      <div className="audit cursor-pointer" role="button" tabIndex={0} onClick={() => flow.start(false)}>
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
      <AuditionNotice flow={flow} />
      {(playerOpen || flow.phase === "ready") && voice.has_audition && (
        <audio
          controls
          autoPlay={flow.phase === "ready"}
          src={flow.takeUrl ?? `/api/voices/${voice.voice_id}/audition.wav?t=0`}
          className="mt-2 h-[30px] w-full"
        />
      )}
    </div>
  );
}
