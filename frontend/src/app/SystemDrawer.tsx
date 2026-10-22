import { useState } from "react";
import { Dialog as AriaDialog, Heading, Modal, ModalOverlay } from "react-aria-components";

import { useEngines, useSystemProbe } from "../api/hooks";
import { SetupChecklist } from "../components/SetupChecklist";
import { HEALTH_LABEL, HEALTH_LED } from "../lib/health";
import { useHealth } from "./useHealth";

/** Right-hand drawer (react-aria modal: focus trap, Esc, click-outside) holding the setup
    checklist. "Check again" is a single manual refetch — never a loop. */
export function SystemDrawer({ onClose }: { onClose: () => void }) {
  const probe = useSystemProbe();
  const engines = useEngines();
  return (
    <ModalOverlay
      isOpen
      isDismissable
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      className="overlay on drawer-ov"
    >
      <Modal className="drawer">
        <AriaDialog className="dbody">
          <div className="dh">
            <Heading slot="title" className="dt">
              System
            </Heading>
            <button className="key quiet" onClick={onClose}>
              esc
            </button>
          </div>
          <div className="db">
            <p className="drawer-sub">What Seiyuu needs on this computer, and how to fix anything that's missing.</p>
            <SetupChecklist />
          </div>
          <div className="df">
            {probe.data && <span className="drawer-ver">seiyuu {probe.data.version}</span>}
            <span className="flex-1" />
            <button
              className="key quiet"
              disabled={probe.isFetching}
              onClick={() => {
                void probe.refetch();
                void engines.refetch();
              }}
            >
              {probe.isFetching ? "checking…" : "check again"}
            </button>
          </div>
        </AriaDialog>
      </Modal>
    </ModalOverlay>
  );
}

/** The NavRail footer's live status pill; opens the System drawer. */
export function HealthPill() {
  const level = useHealth();
  const [open, setOpen] = useState(false);
  return (
    <>
      <button className={`hpill ${level}`} title="System status — open the setup checklist" onClick={() => setOpen(true)}>
        <i className={`led ${HEALTH_LED[level]}`} aria-hidden />
        {HEALTH_LABEL[level]}
      </button>
      {open && <SystemDrawer onClose={() => setOpen(false)} />}
    </>
  );
}
