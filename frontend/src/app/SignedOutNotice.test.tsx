import { act, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { api } from "../api/client";
import { errorResponse, jsonResponse, mockApi } from "../test/utils";
import { SignedOutNotice } from "./SignedOutNotice";

describe("SignedOutNotice", () => {
  it("appears when a request finds the login expired, and goes when one succeeds again", async () => {
    render(<SignedOutNotice />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    mockApi().on("GET", "/api/books", () => errorResponse(401, "signed_out", "Sign in again"));
    await act(async () => {
      await api("/api/books").catch(() => undefined);
    });
    expect(screen.getByRole("alert")).toHaveTextContent("Your sign-in has expired");
    // a real page load, so the browser reaches Cloudflare's login
    expect(screen.getByRole("link", { name: "Sign in again" })).toHaveAttribute("href", "/");

    mockApi().on("GET", "/api/books", () => jsonResponse({ books: [] }));
    await act(async () => {
      await api("/api/books");
    });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
