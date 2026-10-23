import { Button, Menu, MenuItem, MenuTrigger, Popover } from "react-aria-components";

/** The roster's small "View" menu: display preferences that change what the page shows, never
    what the book says. Today one checkable item, spoiler-safe. react-aria gives it the menu
    keyboard model (arrows, Enter/Space, Escape) and menuitemcheckbox semantics. */
export function ViewMenu({
  spoilerSafe,
  onSpoilerSafe,
}: {
  spoilerSafe: boolean;
  onSpoilerSafe: (on: boolean) => void;
}) {
  return (
    <MenuTrigger>
      <Button className="key quiet ml-auto px-[9px] py-[3px]">
        View <span aria-hidden>▾</span>
      </Button>
      <Popover placement="bottom end" offset={2} className="tsel-pop">
        <Menu
          aria-label="view"
          selectionMode="multiple"
          selectedKeys={spoilerSafe ? ["spoiler"] : []}
          onSelectionChange={(keys) => onSpoilerSafe(keys === "all" || keys.has("spoiler"))}
          className="outline-none"
        >
          <MenuItem id="spoiler" textValue="Spoiler-safe" className="tsel-item flex items-start gap-2">
            {({ isSelected }) => (
              <>
                <span aria-hidden className="mono w-3 text-tungsten">
                  {isSelected ? "✓" : ""}
                </span>
                <span>
                  Spoiler-safe
                  <span className="block text-[11px] text-ink-3">
                    hide characters you haven't met yet, here and in every list
                  </span>
                </span>
              </>
            )}
          </MenuItem>
        </Menu>
      </Popover>
    </MenuTrigger>
  );
}
