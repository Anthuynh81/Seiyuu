/** Per-viewer conveniences only (volume, last book, reading prefs). A private window or a
    browser with site data blocked throws on ANY localStorage touch — even reading the
    `localStorage` global — so every access goes through these and simply degrades to
    "not remembered". Nothing the app needs to be correct may live here. */

export function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** Forgets a remembered value (a refused removal just leaves it remembered). */
export function removeStored(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    /* storage blocked: nothing to forget */
  }
}

/** Returns false when the browser refused the write (the value then lasts this page only). */
export function writeStored(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}
