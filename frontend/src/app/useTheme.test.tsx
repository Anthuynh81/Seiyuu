import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { useTheme } from "./useTheme";

type ChangeListener = (ev: MediaQueryListEvent) => void;

/** Controllable matchMedia fake. Must be installed BEFORE rendering useTheme — the hook
    calls matchMedia inside its effect on mount. `setMatches` flips the OS scheme and fires
    the change event the hook subscribes to in system mode. */
function installMatchMedia(initialMatches: boolean) {
  const listeners = new Set<ChangeListener>();
  const fake = {
    matches: initialMatches,
    media: "(prefers-color-scheme: light)",
    onchange: null,
    addEventListener(_type: string, listener: ChangeListener) {
      listeners.add(listener);
    },
    removeEventListener(_type: string, listener: ChangeListener) {
      listeners.delete(listener);
    },
    addListener() {},
    removeListener() {},
    dispatchEvent: () => false,
  };
  window.matchMedia = () => fake as unknown as MediaQueryList;
  return {
    setMatches(matches: boolean) {
      fake.matches = matches;
      for (const listener of [...listeners]) listener({ matches } as MediaQueryListEvent);
    },
    listenerCount: () => listeners.size,
  };
}

const stampedTheme = () => document.documentElement.dataset.theme;

afterEach(() => {
  delete document.documentElement.dataset.theme;
});

describe("useTheme", () => {
  it("a stored 'booth' pref stamps the dark theme on mount", () => {
    localStorage.setItem("seiyuu-theme", "booth");
    installMatchMedia(true); // OS says light — booth must win anyway

    const { result } = renderHook(useTheme);

    expect(result.current.pref).toBe("booth");
    expect(stampedTheme()).toBe("dark");
  });

  it("setPref('daylight') stamps the light theme and persists under seiyuu-theme", () => {
    installMatchMedia(false);
    const { result } = renderHook(useTheme);
    expect(stampedTheme()).toBe("dark"); // system + OS dark

    act(() => result.current.setPref("daylight"));

    expect(stampedTheme()).toBe("light");
    expect(localStorage.getItem("seiyuu-theme")).toBe("daylight");
  });

  it("'system' follows the OS scheme and re-stamps on matchMedia change events", () => {
    const media = installMatchMedia(true);
    const { result } = renderHook(useTheme);

    expect(result.current.pref).toBe("system");
    expect(stampedTheme()).toBe("light");

    act(() => media.setMatches(false));
    expect(stampedTheme()).toBe("dark");

    act(() => media.setMatches(true));
    expect(stampedTheme()).toBe("light");
  });

  it("a garbage stored value falls back to 'system' (and still follows the OS)", () => {
    localStorage.setItem("seiyuu-theme", "neon-disco");
    installMatchMedia(true);

    const { result } = renderHook(useTheme);

    expect(result.current.pref).toBe("system");
    expect(stampedTheme()).toBe("light"); // proves it follows the OS, not a hardcoded default
  });

  it("leaving 'system' unsubscribes from OS changes so they no longer re-stamp", () => {
    const media = installMatchMedia(false);
    const { result } = renderHook(useTheme);
    expect(stampedTheme()).toBe("dark");

    act(() => result.current.setPref("booth"));
    expect(stampedTheme()).toBe("dark");
    expect(media.listenerCount()).toBe(0);

    act(() => media.setMatches(true)); // OS flips to light — booth must not budge
    expect(stampedTheme()).toBe("dark");
    expect(localStorage.getItem("seiyuu-theme")).toBe("booth");
  });
});
