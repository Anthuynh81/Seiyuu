import type { CSSProperties } from "react";
import { Slider as AriaSlider, SliderThumb, SliderTrack } from "react-aria-components";

/** Talkback fader: a hairline track with a tungsten fill and a square thumb — keyboard
    operable (arrows/home/end) via react-aria, unlike the styled native range input. */
export function TalkSlider({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  ariaLabel,
  className,
  style,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  ariaLabel: string;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <AriaSlider
      value={value}
      onChange={onChange}
      minValue={min}
      maxValue={max}
      step={step}
      aria-label={ariaLabel}
      className={`tfader ${className ?? ""}`}
      style={style}
    >
      <SliderTrack className="tfader-track">
        {({ state }) => (
          <>
            <div className="tfader-fill" style={{ width: `${state.getThumbPercent(0) * 100}%` }} />
            <SliderThumb className="tfader-thumb" />
          </>
        )}
      </SliderTrack>
    </AriaSlider>
  );
}
