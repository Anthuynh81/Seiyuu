import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

/** Shared jsdom shims. jsdom implements neither the observer APIs react-aria overlays
    measure with, nor media playback, nor blob URLs — every stub here exists because a
    component under test calls it. Plain prototype/global assignment (NOT vi.stubGlobal)
    so the shims survive `vi.unstubAllGlobals()` in the per-test cleanup below. */

class ObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords(): never[] {
    return [];
  }
}
if (!("ResizeObserver" in globalThis)) {
  (globalThis as Record<string, unknown>).ResizeObserver = ObserverStub;
}
if (!("IntersectionObserver" in globalThis)) {
  (globalThis as Record<string, unknown>).IntersectionObserver = ObserverStub;
}

// useTheme reads the OS scheme through matchMedia; tests that care about `matches` or the
// change event replace this per-test.
if (typeof window.matchMedia !== "function") {
  window.matchMedia = (query: string): MediaQueryList =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener() {},
      removeEventListener() {},
      addListener() {},
      removeListener() {},
      dispatchEvent: () => false,
    }) as MediaQueryList;
}

// react-aria scrolls focused listbox items into view and (un)captures pointers.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function () {};
}
if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false;
  Element.prototype.setPointerCapture = () => {};
  Element.prototype.releasePointerCapture = () => {};
}

// jsdom's HTMLMediaElement throws "Not implemented" on playback; the player and the demo
// buttons only need the calls to succeed. Spy per-test (vi.spyOn) to assert on them.
Object.defineProperty(HTMLMediaElement.prototype, "play", {
  configurable: true,
  writable: true,
  value: function play() {
    return Promise.resolve();
  },
});
Object.defineProperty(HTMLMediaElement.prototype, "pause", {
  configurable: true,
  writable: true,
  value: function pause() {},
});
Object.defineProperty(HTMLMediaElement.prototype, "load", {
  configurable: true,
  writable: true,
  value: function load() {},
});

// Voices' mixer demo player builds blob URLs from audition WAVs.
if (typeof URL.createObjectURL !== "function") {
  let blobSeq = 0;
  URL.createObjectURL = () => `blob:mock-${blobSeq++}`;
  URL.revokeObjectURL = () => {};
}

afterEach(() => {
  cleanup();
  // Theme pref and player volume persist in localStorage; never bleed across tests.
  localStorage.clear();
});
