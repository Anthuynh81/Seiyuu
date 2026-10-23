import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { makeBook, makeEngine, makeJob, makeSystemProbe } from "../test/fixtures";
import { jsonResponse, mockApi, renderWithProviders } from "../test/utils";
import { NavRail } from "./NavRail";

/** NavRail has no <Routes>; this probe proves NavLink actually moved the router. */
function LocationProbe() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}</output>;
}

/** Workflow order: the shelf, the book's numbered steps, then the tools. */
const screenLinks = [
  "Library",
  "1 Speakers & cast",
  "2 Proof & names",
  "3 Produce",
  "4 Listen",
  "Voices",
  "Series",
];

const hrefOf = (name: string) => screen.getByRole("link", { name }).getAttribute("href");

/** Blocked site data: every localStorage touch throws, like a locked-down browser. */
function blockStorage() {
  const deny = () => {
    throw new DOMException("The operation is insecure.", "SecurityError");
  };
  vi.stubGlobal("localStorage", { getItem: deny, setItem: deny, removeItem: deny, clear: deny, key: deny });
}

describe("NavRail", () => {
  it("renders a navigation link for every screen (no running lamp while idle)", () => {
    mockApi();
    renderWithProviders(<NavRail />);
    for (const name of screenLinks) {
      expect(screen.getByRole("link", { name })).toBeInTheDocument();
    }
    expect(screen.queryByTitle("a job is running")).not.toBeInTheDocument();
  });

  it("lists the screens in workflow order: Library, the numbered book steps, then Tools", () => {
    mockApi();
    renderWithProviders(<NavRail />);
    expect(screen.getAllByRole("link").map((a) => a.textContent?.trim())).toEqual(screenLinks);

    const steps = screen.getByRole("group", { name: "Book steps" });
    expect(within(steps).getAllByRole("link").map((a) => a.getAttribute("href"))).toEqual([
      "/review",
      "/lexicon",
      "/render",
      "/listen",
    ]);
    const tools = screen.getByRole("group", { name: "Tools" });
    expect(within(tools).getAllByRole("link").map((a) => a.textContent)).toEqual(["Voices", "Series"]);
    expect(within(tools).getAllByRole("link").map((a) => a.getAttribute("href"))).toEqual(["/voices", "/series"]);
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

    await user.click(screen.getByRole("link", { name: "Voices" }));
    expect(screen.getByTestId("location").textContent).toBe("/voices");

    await user.click(screen.getByRole("link", { name: "Library" }));
    expect(screen.getByTestId("location").textContent).toBe("/");
  });

  it("theme switch stamps data-theme on <html> and persists the pref under seiyuu-theme", async () => {
    const user = userEvent.setup();
    mockApi();
    renderWithProviders(<NavRail />);

    expect(screen.getByRole("button", { name: "auto" })).toHaveClass("on");
    await user.click(screen.getByRole("button", { name: "light" }));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem("seiyuu-theme")).toBe("daylight");

    await user.click(screen.getByRole("button", { name: "dark" }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("seiyuu-theme")).toBe("booth");
  });

  it("shows the running lamp on the Produce link while a job is running", async () => {
    mockApi().get("/api/jobs", { jobs: [makeJob({ state: "running" })] });
    renderWithProviders(<NavRail />);
    const lamp = await screen.findByTitle("a job is running");
    expect(lamp.closest("a")).toHaveAttribute("href", "/render");
  });

  describe("current book", () => {
    const shelf = () =>
      mockApi().get("/api/books", {
        books: [
          makeBook({ book_id: "heart", title: "The Expired Heart" }),
          makeBook({ book_id: "ghost", title: "Ghost Story" }),
        ],
      });

    it("with no book chosen yet, step links are bare and the shelf isn't fetched", () => {
      const server = mockApi();
      renderWithProviders(<NavRail />);
      expect(hrefOf("3 Produce")).toBe("/render");
      expect(server.calls.some((c) => c.url === "/api/books")).toBe(false);
    });

    it("an explicit ?book= reaches every step link, becomes the remembered book, and never touches tools", async () => {
      shelf();
      renderWithProviders(<NavRail />, { route: "/review?book=ghost" });
      expect(hrefOf("1 Speakers & cast")).toBe("/review?book=ghost");
      expect(hrefOf("2 Proof & names")).toBe("/lexicon?book=ghost");
      expect(hrefOf("3 Produce")).toBe("/render?book=ghost");
      expect(hrefOf("4 Listen")).toBe("/listen?book=ghost");
      expect(hrefOf("Voices")).toBe("/voices");
      expect(hrefOf("Series")).toBe("/series");
      expect(hrefOf("Library")).toBe("/");
      expect(localStorage.getItem("seiyuu.currentBook")).toBe("ghost");
      // the book is named plainly above its steps
      const steps = screen.getByRole("group", { name: "Book steps" });
      expect(await within(steps).findByText("Ghost Story")).toBeInTheDocument();
    });

    it("on a screen without ?book=, the step links carry the remembered book once the shelf confirms it", async () => {
      localStorage.setItem("seiyuu.currentBook", "heart");
      shelf();
      renderWithProviders(<NavRail />, { route: "/voices" });
      await waitFor(() => expect(hrefOf("3 Produce")).toBe("/render?book=heart"));
      expect(screen.getByText("The Expired Heart")).toBeInTheDocument();
    });

    it("an explicit ?book= wins over the remembered one and replaces it", async () => {
      localStorage.setItem("seiyuu.currentBook", "heart");
      shelf();
      renderWithProviders(<NavRail />, { route: "/render?book=ghost" });
      expect(hrefOf("1 Speakers & cast")).toBe("/review?book=ghost");
      await waitFor(() => expect(localStorage.getItem("seiyuu.currentBook")).toBe("ghost"));
      expect(await screen.findByText("Ghost Story")).toBeInTheDocument();
      expect(hrefOf("4 Listen")).toBe("/listen?book=ghost");
    });

    it("a remembered book that has since been deleted is ignored", async () => {
      localStorage.setItem("seiyuu.currentBook", "gone");
      const server = shelf();
      const { queryClient } = renderWithProviders(<NavRail />, { route: "/voices" });
      await waitFor(() => expect(queryClient.getQueryState(["books"])?.status).toBe("success"));
      expect(server.calls.some((c) => c.url === "/api/books")).toBe(true);
      expect(hrefOf("3 Produce")).toBe("/render");
      expect(hrefOf("4 Listen")).toBe("/listen");
      expect(screen.queryByText(/Expired Heart|Ghost Story/)).not.toBeInTheDocument();
    });

    it("following a step link keeps the book in the URL, and leaving for a tool keeps it remembered", async () => {
      const user = userEvent.setup();
      shelf();
      function Probe() {
        const location = useLocation();
        return <output data-testid="where">{location.pathname + location.search}</output>;
      }
      renderWithProviders(
        <>
          <NavRail />
          <Probe />
        </>,
        { route: "/review?book=heart" },
      );
      await user.click(screen.getByRole("link", { name: "3 Produce" }));
      expect(screen.getByTestId("where").textContent).toBe("/render?book=heart");
      await user.click(screen.getByRole("link", { name: "Voices" }));
      expect(screen.getByTestId("where").textContent).toBe("/voices");
      await waitFor(() => expect(hrefOf("4 Listen")).toBe("/listen?book=heart"));
    });

    it("blocked storage doesn't crash the rail; an explicit ?book= is remembered for this page only", async () => {
      const user = userEvent.setup();
      blockStorage();
      expect(() => localStorage.getItem("seiyuu.currentBook")).toThrow();
      shelf();
      renderWithProviders(<NavRail />, { route: "/review?book=heart" });
      expect(hrefOf("3 Produce")).toBe("/render?book=heart");
      await user.click(screen.getByRole("link", { name: "Voices" }));
      await waitFor(() => expect(hrefOf("3 Produce")).toBe("/render?book=heart"));
      // the theme switch still works, just unremembered
      await user.click(screen.getByRole("button", { name: "light" }));
      expect(document.documentElement.dataset.theme).toBe("light");
    });
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
