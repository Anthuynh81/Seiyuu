import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { makeJob } from "../test/fixtures";
import { mockApi, renderWithProviders } from "../test/utils";
import { NavRail } from "./NavRail";

/** NavRail has no <Routes>; this probe proves NavLink actually moved the router. */
function LocationProbe() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}</output>;
}

const screenLinks = [
  "Library",
  "Listen",
  "Character Review",
  "Pronunciation",
  "Voice Studio",
  "Series",
  "Render & Jobs",
];

describe("NavRail", () => {
  it("renders a navigation link for every screen (no running lamp while idle)", () => {
    mockApi();
    renderWithProviders(<NavRail />);
    for (const name of screenLinks) {
      expect(screen.getByRole("link", { name })).toBeInTheDocument();
    }
    expect(screen.queryByTitle("a job is running")).not.toBeInTheDocument();
  });

  it("clicking a nav link navigates the router to that screen's route", async () => {
    const user = userEvent.setup();
    mockApi();
    renderWithProviders(
      <>
        <NavRail />
        <LocationProbe />
      </>,
      { route: "/render" },
    );

    await user.click(screen.getByRole("link", { name: "Voice Studio" }));
    expect(screen.getByTestId("location").textContent).toBe("/voices");

    await user.click(screen.getByRole("link", { name: "Library" }));
    expect(screen.getByTestId("location").textContent).toBe("/");
  });

  it("theme switch stamps data-theme on <html> and persists the pref under seiyuu-theme", async () => {
    const user = userEvent.setup();
    mockApi();
    renderWithProviders(<NavRail />);

    await user.click(screen.getByRole("button", { name: "day" }));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem("seiyuu-theme")).toBe("daylight");

    await user.click(screen.getByRole("button", { name: "booth" }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("seiyuu-theme")).toBe("booth");
  });

  it("shows the running lamp on the Render & Jobs link while a job is running", async () => {
    mockApi().get("/api/jobs", { jobs: [makeJob({ state: "running" })] });
    renderWithProviders(<NavRail />);
    expect(await screen.findByTitle("a job is running")).toBeInTheDocument();
  });
});
