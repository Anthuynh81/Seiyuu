import { describe, expect, it, vi } from "vitest";

import { pickCurrentBook, readRememberedBook, rememberBook, withBook } from "./currentBook";

describe("pickCurrentBook", () => {
  it("an explicit ?book= always wins, even before the shelf loads", () => {
    expect(pickCurrentBook("b", "a", ["a", "b"])).toBe("b");
    expect(pickCurrentBook("b", "a", undefined)).toBe("b");
  });

  it("uses the remembered id only once the shelf confirms the book still exists", () => {
    expect(pickCurrentBook(null, "a", ["a", "b"])).toBe("a");
    expect(pickCurrentBook(null, "a", undefined)).toBeNull(); // shelf still loading
    expect(pickCurrentBook(null, "gone", ["a", "b"])).toBeNull(); // deleted since
    expect(pickCurrentBook(null, null, ["a"])).toBeNull();
  });
});

describe("withBook", () => {
  it("adds an encoded ?book= only when there is a book", () => {
    expect(withBook("/render", "a b&c")).toBe("/render?book=a%20b%26c");
    expect(withBook("/render", null)).toBe("/render");
  });
});

describe("remembering", () => {
  it("round-trips through localStorage", () => {
    rememberBook("heart");
    expect(localStorage.getItem("seiyuu.currentBook")).toBe("heart");
    expect(readRememberedBook()).toBe("heart");
  });

  it("blocked storage never throws: the choice lasts for this page only", () => {
    const deny = () => {
      throw new DOMException("denied", "SecurityError");
    };
    vi.stubGlobal("localStorage", { getItem: deny, setItem: deny, removeItem: deny, clear: deny });
    expect(readRememberedBook()).toBeNull();
    expect(() => rememberBook("heart")).not.toThrow();
    expect(readRememberedBook()).toBe("heart");
  });

  it("a write that fails while reads still work (full origin) remembers the new book, not the stale one", () => {
    localStorage.setItem("seiyuu.currentBook", "old");
    const store = localStorage;
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => store.getItem(k),
      setItem: () => {
        throw new DOMException("full", "QuotaExceededError");
      },
    });
    expect(() => rememberBook("heart")).not.toThrow();
    expect(readRememberedBook()).toBe("heart");
  });
});
