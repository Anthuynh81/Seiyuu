import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { makeEngine, makeJob, makeSystemProbe } from "../test/fixtures";
import { jsonResponse, mockApi, renderWithProviders } from "../test/utils";
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

  describe("health pill", () => {
    const healthy = () =>
      mockApi().get("/api/system?probe=true", makeSystemProbe()).get("/api/engines", { engines: [makeEngine()] });

    it("replaces the hard-coded footer with a live Ready pill when every required check passes", async () => {
      healthy();
      renderWithProviders(<NavRail />);
      expect(await screen.findByRole("button", { name: "Ready" })).toBeInTheDocument();
      expect(screen.queryByText(/console online/)).not.toBeInTheDocument();
    });

    it("says Needs setup when Ollama isn't running", async () => {
      healthy().get(
        "/api/system?probe=true",
        makeSystemProbe({ ollama: { base_url: "http://localhost:11434", reachable: false, model_installed: null } }),
      );
      renderWithProviders(<NavRail />);
      expect(await screen.findByRole("button", { name: "Needs setup" })).toBeInTheDocument();
    });

    it("says Backend offline when the Vite proxy answers an empty 500", async () => {
      mockApi().on("GET", "/api/system", () => new Response("", { status: 500 }));
      renderWithProviders(<NavRail />);
      expect(await screen.findByRole("button", { name: "Backend offline" })).toBeInTheDocument();
    });

    it("says Backend offline on a network error", async () => {
      vi.stubGlobal("fetch", () => Promise.reject(new TypeError("Failed to fetch")));
      renderWithProviders(<NavRail />);
      expect(await screen.findByRole("button", { name: "Backend offline" })).toBeInTheDocument();
    });

    it("up, then down: a failed refetch flips Ready to Backend offline despite the cached probe", async () => {
      const user = userEvent.setup();
      const server = healthy();
      const { queryClient } = renderWithProviders(<NavRail />);
      expect(await screen.findByRole("button", { name: "Ready" })).toBeInTheDocument();

      // uvicorn stops: the Vite proxy answers an empty 502 for everything
      server.on("GET", /^\/api\//, () => new Response("", { status: 502 }));
      await queryClient.refetchQueries({ queryKey: ["system", "probe"] });

      await user.click(await screen.findByRole("button", { name: "Backend offline" }));
      const drawer = await screen.findByRole("dialog", { name: "System" });
      const backend = within(drawer).getByRole("listitem", { name: "Seiyuu backend" });
      expect(within(backend).getByText("Not running")).toBeInTheDocument();
      expect(within(drawer).getAllByRole("listitem")).toHaveLength(1); // no stale greens
    });

    it("a running backend answering with an error is Needs setup, not Backend offline", async () => {
      mockApi().error("GET", "/api/system", 503, "lock_timeout", "database is locked");
      renderWithProviders(<NavRail />);
      expect(await screen.findByRole("button", { name: "Needs setup" })).toBeInTheDocument();
    });

    it("when the backend comes back, the engine catalog is refetched once", async () => {
      const server = mockApi().on("GET", /^\/api\/(system|engines)/, () => new Response("", { status: 502 }));
      const { queryClient } = renderWithProviders(<NavRail />);
      expect(await screen.findByRole("button", { name: "Backend offline" })).toBeInTheDocument();
      const enginesCalls = () => server.calls.filter((c) => c.url === "/api/engines").length;
      const before = enginesCalls();

      server.get("/api/system?probe=true", makeSystemProbe()).get("/api/engines", { engines: [makeEngine()] });
      await queryClient.refetchQueries({ queryKey: ["system", "probe"] });

      expect(await screen.findByRole("button", { name: "Ready" })).toBeInTheDocument();
      await waitFor(() => expect(enginesCalls()).toBe(before + 1));
      expect(queryClient.getQueryState(["engines"])?.status).toBe("success");
    });

    it("shows Checking… until the probe answers", () => {
      mockApi().on("GET", "/api/system", () => new Promise<Response>(() => {}));
      renderWithProviders(<NavRail />);
      expect(screen.getByRole("button", { name: "Checking…" })).toBeInTheDocument();
    });

    it("opens the System drawer with the checklist; esc closes it", async () => {
      const user = userEvent.setup();
      healthy().get(
        "/api/system?probe=true",
        makeSystemProbe({ ollama: { base_url: "x", reachable: true, model_installed: false } }),
      );
      renderWithProviders(<NavRail />);
      await user.click(await screen.findByRole("button", { name: "Needs setup" }));

      const drawer = await screen.findByRole("dialog", { name: "System" });
      const list = within(drawer).getByRole("list", { name: "Setup checklist" });
      expect(within(list).getByText("ollama pull qwen2.5:7b")).toBeInTheDocument();
      expect(within(drawer).getByText("seiyuu 0.9.0")).toBeInTheDocument();

      await user.click(within(drawer).getByRole("button", { name: "esc" }));
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("check again refetches the probe once (a manual refresh, not a loop)", async () => {
      const user = userEvent.setup();
      const server = healthy();
      renderWithProviders(<NavRail />);
      await user.click(await screen.findByRole("button", { name: "Ready" }));
      const drawer = await screen.findByRole("dialog", { name: "System" });
      const before = server.calls.filter((c) => c.url === "/api/system?probe=true").length;
      server.on("GET", "/api/system?probe=true", () => jsonResponse(makeSystemProbe({ ffmpeg_available: false })));
      await user.click(within(drawer).getByRole("button", { name: "check again" }));
      expect(await within(drawer).findByText("Not found")).toBeInTheDocument(); // the ffmpeg row
      const after = server.calls.filter((c) => c.url === "/api/system?probe=true").length;
      expect(after).toBe(before + 1);
    });
  });
});
