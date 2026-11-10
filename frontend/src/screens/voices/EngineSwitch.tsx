import { useBooks, useSwitchVoiceEngine } from "../../api/hooks";
import type { VoiceOut } from "../../api/types";
import { PlainError } from "../../components/PlainError";
import { switchTarget } from "./helpers";

const SHORT_NAME: Record<string, string> = { breeze: "Breeze", qwen3tts: "Qwen" };

/** "use Breeze" / "use Qwen" on a cloned voice's card: the same voice (same clip, consent and
    casting) cloned by the other engine from now on. Nothing recorded is deleted; the reply
    names the books to record again, shown here by title. */
export function EngineSwitch({ voice }: { voice: VoiceOut }) {
  const move = useSwitchVoiceEngine();
  const books = useBooks();
  const target = switchTarget(voice);
  const done = move.data && move.data.voice_id === voice.voice_id ? move.data : null;
  if (target === null && done === null) return null;
  const titleOf = (id: string) => books.data?.books.find((b) => b.book_id === id)?.title ?? id;
  const name = (engine: string) => SHORT_NAME[engine] ?? engine;
  return (
    <div className="mt-1 text-xs">
      {target !== null && (
        <button
          className="link"
          disabled={move.isPending}
          title={`Clone this voice with ${name(target)} from now on. Its clip, consent and casting stay the same, and lines already recorded are kept.`}
          onClick={() => move.mutate({ voiceId: voice.voice_id, engine: target })}
        >
          {move.isPending ? "switching…" : `use ${name(target)}`}
        </button>
      )}
      {done && (
        <p role="status" className="m-0 mt-1 text-ink-2">
          Now cloned by {name(done.engine)} (was {name(done.previous_engine)}).{" "}
          {done.books.length > 0
            ? `Record ${done.books.length === 1 ? "this book" : "these books"} again to hear it: ${done.books.map(titleOf).join(", ")}.`
            : "No book uses it yet."}
        </p>
      )}
      {move.error && <PlainError className="mt-1" error={move.error} title="couldn't switch engine" />}
    </div>
  );
}
