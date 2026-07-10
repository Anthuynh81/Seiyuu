import type { ReactNode } from "react";
import { Focusable, OverlayArrow, Tooltip as AriaTooltip, TooltipTrigger } from "react-aria-components";

/** Talkback tooltip: a raised console chip instead of the OS title bubble — keyboard
    focus shows it too. Wrap exactly one focusable child. */
export function Tip({ content, children, delay = 400 }: { content: ReactNode; children: ReactNode; delay?: number }) {
  return (
    <TooltipTrigger delay={delay}>
      <Focusable>{children as never}</Focusable>
      <AriaTooltip offset={6} className="ttip">
        <OverlayArrow>
          <span className="ttip-arrow" aria-hidden />
        </OverlayArrow>
        {content}
      </AriaTooltip>
    </TooltipTrigger>
  );
}
