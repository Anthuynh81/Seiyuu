import { useRef, useState } from "react";
import { Button, Menu, MenuItem, MenuTrigger, Popover } from "react-aria-components";
import { useNavigate } from "react-router-dom";

import type { SegmentRow } from "../../api/types";
import { lexiconHref, pronunciationTerm, reviewChapterHref, reviewLineHref } from "../../lib/listen";

/** The speaker chip in Listen's margin, as a menu back to the fixes: change who says this line
    (Review, landed on it), fix how a word is said (Pronunciation, the selected or last clicked
    word waiting in a new row), or open the chapter in Review. react-aria gives it the menu
    keyboard model (arrows, Enter/Space, Escape). Navigation goes through the router so the
    player keeps playing. */
export function LineMenu({
  row,
  bookId,
  chapter,
  label,
  pickedWord,
}: {
  row: SegmentRow;
  bookId: string;
  chapter: number;
  /** the chip text: the speaker's name, or "narration" */
  label: string;
  /** the word last clicked in this line, if any (the fallback when nothing is selected) */
  pickedWord: () => string | null;
}) {
  const navigate = useNavigate();
  // The text selection is read on pointer/key DOWN: by the time the menu opens, pressing the
  // chip may already have collapsed it.
  const selected = useRef<string | null>(null);
  const [term, setTerm] = useState<string | null>(null);
  const segKey = `${row.block_id}:${row.segment_index}`;

  const readSelection = () => {
    try {
      const sel = window.getSelection();
      const node = sel?.anchorNode ?? null;
      const el = node instanceof Element ? node : node?.parentElement;
      const inLine = el?.closest("[data-seg]")?.getAttribute("data-seg") === segKey;
      selected.current = inLine ? pronunciationTerm(sel?.toString()) : null;
    } catch {
      selected.current = null;
    }
  };

  const who = row.speaker === null ? "the narrator" : label;
  return (
    <span onPointerDownCapture={readSelection} onKeyDownCapture={readSelection}>
      <MenuTrigger
        onOpenChange={(open) => {
          if (open) setTerm(selected.current ?? pronunciationTerm(pickedWord()));
        }}
      >
        <Button className={`chip ${row.speaker === null ? "narr" : ""}`} aria-label={`${label}: fix this line`}>
          {row.speaker === null ? label : label.toUpperCase()}
        </Button>
        <Popover placement="bottom start" offset={2} className="tsel-pop">
          <Menu
            aria-label="fix this line"
            className="outline-none"
            onAction={(id) => {
              if (id === "line") navigate(reviewLineHref(bookId, chapter, row));
              else if (id === "say") navigate(lexiconHref(bookId, term));
              else if (id === "review") navigate(reviewChapterHref(bookId, chapter));
            }}
          >
            <MenuItem id="line" textValue="Fix this line" className="tsel-item">
              Fix this line
              <span className="block text-[11px] text-ink-3">change who says it — now {who}</span>
            </MenuItem>
            <MenuItem id="say" textValue="Fix pronunciation" className="tsel-item">
              {term ? `Fix pronunciation of “${term}”` : "Fix pronunciation"}
              <span className="block text-[11px] text-ink-3">
                {term ? "add how it should sound" : "select a word first to fill it in"}
              </span>
            </MenuItem>
            <MenuItem id="review" textValue="Open in Review" className="tsel-item">
              Open in Review
              <span className="block text-[11px] text-ink-3">this chapter, every line</span>
            </MenuItem>
          </Menu>
        </Popover>
      </MenuTrigger>
    </span>
  );
}
