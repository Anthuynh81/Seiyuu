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

export interface ClipWord {
  el: HTMLElement;
  offset: number; // seconds from the clip start (when this word begins)
  end: number; // seconds from the clip start (when this word ends / the next begins)
}

/** Build word spans across all of a clip's rows, distributing the clip's duration
    over the combined weighted text. Offsets are seconds from the clip start; `end` is
    the next word's offset (last word ends at the clip duration). This is the FALLBACK
    used while whisper words are still loading or on a 404. */
export function buildClipWords(rows: { text: string; el: HTMLElement }[], duration: number): ClipWord[] {
  const perRow = rows.map((r) => ({ el: r.el, parts: r.text.split(/\s+/).filter(Boolean) }));
  const totalWeight = perRow.reduce((a, r) => a + r.parts.reduce((x, w) => x + wordWeight(w), 0), 0) || 1;
  const words: ClipWord[] = [];
  let t = 0;
  for (const row of perRow) {
    row.el.textContent = "";
    for (const part of row.parts) {
      const span = document.createElement("span");
      span.className = "w";
      span.textContent = part;
      row.el.appendChild(span);
      row.el.appendChild(document.createTextNode(" "));
      words.push({ el: span, offset: t, end: duration });
      t += (duration * wordWeight(part)) / totalWeight;
    }
  }
  // each word ends where the next begins; the last already ends at `duration`
  for (let i = 0; i < words.length - 1; i++) words[i].end = words[i + 1].offset;
  return words;
}

/* -------------------------------------------------- whisper → source alignment */

export interface WhisperWord {
  start: number;
  end: number;
  word: string;
}

export interface WordSpan {
  offset: number;
  end: number;
}

/** Fold a token to its comparison key: lowercase, curly→straight quotes, then drop every
    non-alphanumeric char. This makes `"Hello,"`, `hello`, and `“hello”` compare equal, and
    `1,000` compare equal to `1000`. Pure digits and pure words never coincidentally match
    (different chars survive), so genuine mismatches still fall to interpolation. */
export function foldToken(t: string): string {
  return t
    .toLowerCase()
    .replace(/[‘’‛′]/g, "'")
    .replace(/[“”″]/g, '"')
    .replace(/[^a-z0-9]/g, "");
}

/** Longest-common-subsequence anchors between two folded token lists. Returns [display,
    whisper] index pairs in order. Empty folded tokens never match (pure-punctuation tokens
    are treated as unmatched and interpolated). */
function lcsAnchors(a: string[], b: string[]): [number, number][] {
  const n = a.length;
  const m = b.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] !== "" && a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const pairs: [number, number][] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] !== "" && a[i] === b[j]) {
      pairs.push([i, j]);
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      i++;
    } else {
      j++;
    }
  }
  return pairs;
}

/** Map whisper's SPOKEN tokens onto the DISPLAYED source tokens and return an (offset,end)
    per display token, in seconds within the clip. Matched tokens take their whisper time
    verbatim; unmatched runs are gap-interpolated between the surrounding anchors (or the
    clip edges), and times are clamped monotonic non-decreasing inside [0, duration].
    Whisper may be shorter (extra display tokens interpolated) or longer (extra spoken
    tokens ignored). Returns one entry per display token, always. */
export function alignWordTimings(
  displayTokens: string[],
  whisper: WhisperWord[],
  duration: number,
): WordSpan[] {
  const n = displayTokens.length;
  if (n === 0) return [];
  const clipDur = duration > 0 ? duration : whisper.length ? whisper[whisper.length - 1].end : 0;

  // no usable whisper timing: fall back to an even spread so the caller still gets n spans
  if (whisper.length === 0) {
    return displayTokens.map((_, i) => ({ offset: (clipDur * i) / n, end: (clipDur * (i + 1)) / n }));
  }

  const anchors = lcsAnchors(displayTokens.map(foldToken), whisper.map((w) => foldToken(w.word)));
  const spans: (WordSpan | null)[] = new Array<WordSpan | null>(n).fill(null);
  for (const [di, wj] of anchors) spans[di] = { offset: whisper[wj].start, end: whisper[wj].end };

  // interpolate every unmatched run between its bracketing anchors (or the clip edges)
  let i = 0;
  while (i < n) {
    if (spans[i]) {
      i++;
      continue;
    }
    let j = i;
    while (j < n && !spans[j]) j++;
    const startBound = i > 0 ? spans[i - 1]!.end : 0;
    const endBound = j < n ? spans[j]!.offset : clipDur;
    const span = Math.max(0, endBound - startBound);
    const k = j - i;
    for (let r = 0; r < k; r++) {
      spans[i + r] = {
        offset: startBound + (span * r) / k,
        end: startBound + (span * (r + 1)) / k,
      };
    }
    i = j;
  }

  // clamp monotonic non-decreasing and inside the clip
  let prev = 0;
  const out = spans as WordSpan[];
  for (const s of out) {
    s.offset = Math.min(Math.max(s.offset, prev), clipDur);
    s.end = Math.min(Math.max(s.end, s.offset), clipDur);
    prev = s.offset;
  }
  return out;
}
