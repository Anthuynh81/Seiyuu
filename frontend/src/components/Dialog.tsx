import type { ReactNode } from "react";
import { Dialog as AriaDialog, Heading, Modal, ModalOverlay } from "react-aria-components";

/** Talkback modal: focus trap, Esc, and click-outside from react-aria; the .overlay/.dialog
    skin (raised surface, strong hairline, square corners) from the component layer. Header
    and footer are part of the grammar — every dialog names itself and ends in keys. */
export function TalkDialog({
  title,
  onClose,
  dismissable = true,
  children,
  footer,
}: {
  title: ReactNode;
  /** omit to make the dialog non-dismissable (no esc key, no click-outside) */
  onClose?: () => void;
  dismissable?: boolean;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const canDismiss = dismissable && onClose !== undefined;
  return (
    <ModalOverlay
      isOpen
      isDismissable={canDismiss}
      isKeyboardDismissDisabled={!canDismiss}
      onOpenChange={(open) => {
        if (!open) onClose?.();
      }}
      className="overlay on"
    >
      <Modal className="dialog">
        <AriaDialog className="dbody">
          <div className="dh">
            <Heading slot="title" className="dt">
              {title}
            </Heading>
            {canDismiss && (
              <button className="key quiet" onClick={onClose}>
                esc
              </button>
            )}
          </div>
          <div className="db">{children}</div>
          {footer && <div className="df">{footer}</div>}
        </AriaDialog>
      </Modal>
    </ModalOverlay>
  );
}
