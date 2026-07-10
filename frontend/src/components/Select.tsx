import { Button, ListBox, ListBoxItem, Popover, Select as AriaSelect, SelectValue } from "react-aria-components";

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

/** Talkback select: react-aria for keyboard/aria/typeahead, our console skin for the
    popover (the OS-native dropdown ignores the theme entirely). API mirrors a native
    select bound to string values. */
export function TalkSelect({
  value,
  onChange,
  options,
  ariaLabel,
  className,
  popClassName,
}: {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  ariaLabel: string;
  /** extra classes for the trigger button (e.g. "bookpick" for the screen-title picker) */
  className?: string;
  popClassName?: string;
}) {
  return (
    <AriaSelect
      selectedKey={value}
      onSelectionChange={(key) => {
        if (key !== null) onChange(String(key));
      }}
      aria-label={ariaLabel}
    >
      <Button className={`tsel ${className ?? ""}`}>
        <SelectValue className="tsel-val" />
        <span aria-hidden className="tsel-caret">
          ▾
        </span>
      </Button>
      <Popover offset={2} className={`tsel-pop ${popClassName ?? ""}`}>
        <ListBox>
          {options.map((o) => (
            <ListBoxItem key={o.value} id={o.value} isDisabled={o.disabled} textValue={o.label} className="tsel-item">
              {o.label}
            </ListBoxItem>
          ))}
        </ListBox>
      </Popover>
    </AriaSelect>
  );
}
