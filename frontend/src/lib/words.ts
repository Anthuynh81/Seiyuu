/** Read-along timing: rows sharing one rendered wav collapse into one clip, and each
    clip's duration is distributed over its words by weighted interpolation. This is
    the logic that was wrong twice in the field (per-segment clips duplicating
    single-voice audio; unweighted offsets drifting) — keep it pure and tested. */

interface RowLike {
  block_id: string;
  segment_index: number;
  audio_segment: number | null;
  duration_seconds: number | null;
  speaker: string | null;
  speaker_name: string | null;
  text: string;
}

export interface RowGroup<R extends RowLike> {
  key: string; // `${block_id}:${audio_segment}` — one rendered wav
  duration: number;
  speaker: string;
  rows: R[];
}

/** Consecutive rows with the same (block_id, audio_segment) share ONE wav: a
    multivoice render maps rows 1:1, a single-voice render gives a whole block's rows
    the same clip. Never emit the same wav twice. */
export function groupPlayableRows<R extends RowLike>(rows: R[]): RowGroup<R>[] {
  const groups: RowGroup<R>[] = [];
  for (const row of rows) {
    const key = `${row.block_id}:${row.audio_segment}`;
    const last = groups[groups.length - 1];
    if (last && last.key === key) last.rows.push(row);
    else
      groups.push({
        key,
        duration: row.duration_seconds!,
        speaker: row.speaker_name ?? (row.speaker === null ? "narration" : row.speaker),
        rows: [row],
      });
  }
  return groups;
}

/** TTS lingers on clause and sentence boundaries — weight those tokens heavier so the
    interpolated word offsets track the narration much more closely. */
export function wordWeight(w: string): number {
  let extra = 0;
  if (/[.!?…]["”']?$/.test(w)) extra = 5;
  else if (/[;:—]["”']?$/.test(w)) extra = 3;
  else if (/,["”']?$/.test(w)) extra = 2;
  return w.length + 1 + extra;
}

/** Build word spans across all of a clip's rows, distributing the clip's duration
    over the combined weighted text. Offsets are seconds from the clip start. */
export function buildClipWords(rows: { text: string; el: HTMLElement }[], duration: number) {
  const perRow = rows.map((r) => ({ el: r.el, parts: r.text.split(/\s+/).filter(Boolean) }));
  const totalWeight = perRow.reduce((a, r) => a + r.parts.reduce((x, w) => x + wordWeight(w), 0), 0) || 1;
  const words: { el: HTMLElement; offset: number }[] = [];
  let t = 0;
  for (const row of perRow) {
    row.el.textContent = "";
    for (const part of row.parts) {
      const span = document.createElement("span");
      span.className = "w";
      span.textContent = part;
      row.el.appendChild(span);
      row.el.appendChild(document.createTextNode(" "));
      words.push({ el: span, offset: t });
      t += (duration * wordWeight(part)) / totalWeight;
    }
  }
  return words;
}
