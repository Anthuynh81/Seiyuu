import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { CharactersOverview, SystemProbeOut } from "../../api/types";
import { makeJob, makeSystemProbe } from "../../test/fixtures";
import { errorResponse, jsonResponse, mockApi, renderWithProviders, type MockApi } from "../../test/utils";
import { AttributePanel, type AttributePanelBook } from "./AttributePanel";

const fresh: AttributePanelBook = { book_id: "demo", attributed: false, active_job: null };
const found: AttributePanelBook = { book_id: "demo", attributed: true, active_job: null };

/** /api/system and /api/system?probe=true both answer with the probe payload (a superset). */
function routes(probe: Partial<SystemProbeOut> = {}): MockApi {
  return mockApi().get(
    "/api/system",
    makeSystemProbe({ keys: { anthropic_configured: true, elevenlabs_configured: false }, ...probe }),
  );
}

function overview(over: Partial<CharactersOverview> = {}): CharactersOverview {
  return {
    book_id: "demo",
    provider_id: "local",
    model_id: "qwen2.5:7b",
    prompt_version: "v6",
    narration_segments: 10,
    low_confidence_segments: 0,
    unattributed_quote_segments: 0,
    confidence_threshold: 0.7,
    characters: [],
    flagged: [],
    notes: [],
    edit_warnings: [],
    ...over,
  };
}

/** Records every attribute POST body; 402 unless confirm_paid when `paid`. */
function recordAttribute(server: MockApi, { paid = false } = {}) {
  const bodies: Record<string, unknown>[] = [];
  server.on("POST", "/api/books/demo/attribute", (_url, init) => {
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    bodies.push(body);
    if (paid && !body.confirm_paid) {
      return errorResponse(402, "payment_confirmation_required", "attribution with provider=anthropic is paid");
    }
    return jsonResponse(makeJob({ kind: "attribute", state: "queued" }), 202);
  });
  return bodies;
}

const startKey = () => screen.findByRole("button", { name: "▶ find the speakers in the whole book" });

describe("AttributePanel — first run", () => {
  it("starts the whole book with the server's default reader, then says where to follow it", async () => {
    const user = userEvent.setup();
    const server = routes();
    const bodies = recordAttribute(server);
    renderWithProviders(<AttributePanel book={fresh} />);

    expect(await screen.findByRole("region", { name: "Full cast: find the speakers" })).toBeInTheDocument();
    await screen.findByText(/this PC \(Ollama\) · qwen2.5:7b/);
    await user.click(await startKey());
    await waitFor(() => expect(bodies).toEqual([{ chapters: [], provider: "local", model: "qwen2.5:7b" }]));
    expect(await screen.findByText(/Started\. Follow it in the bar at the bottom/)).toBeInTheDocument();
    expect(await startKey()).toBeDisabled(); // no double enqueue
  });

  it("is disabled with one plain reason and the fix when the probe says Ollama is down", async () => {
    const server = routes({ ollama: { base_url: "x", reachable: false, model_installed: null } });
    const bodies = recordAttribute(server);
    renderWithProviders(<AttributePanel book={fresh} />);

    const key = await startKey();
    await waitFor(() => expect(key).toBeDisabled());
    expect(key).toHaveAccessibleDescription(/Ollama isn't running/);
    expect(screen.getByText("ollama serve")).toBeInTheDocument();
    expect(screen.getAllByText(/isn't running|isn't downloaded/)).toHaveLength(1); // ONE reason
    expect(bodies).toHaveLength(0);
  });

  it("names the pull command when the configured model isn't downloaded", async () => {
    routes({ ollama: { base_url: "x", reachable: true, model_installed: false } });
    renderWithProviders(<AttributePanel book={fresh} />);
    const key = await startKey();
    await waitFor(() => expect(key).toBeDisabled());
    expect(screen.getByText("ollama pull qwen2.5:7b")).toBeInTheDocument();
  });

  it("switching to Claude lifts the Ollama block; the paid 402 needs the explicit confirm", async () => {
    const user = userEvent.setup();
    const server = routes({ ollama: { base_url: "x", reachable: false, model_installed: null } });
    const bodies = recordAttribute(server, { paid: true });
    renderWithProviders(<AttributePanel book={fresh} />);

    await user.click(await screen.findByRole("button", { name: "change" }));
    await user.click(screen.getByRole("button", { name: /speaker-finding provider/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: /Claude online, paid/ }));
    const key = await startKey();
    await waitFor(() => expect(key).toBeEnabled());
    expect(screen.queryByText("ollama serve")).not.toBeInTheDocument();

    await user.click(key);
    // plain words first; the server's text stays as secondary detail
    expect(await screen.findByText(/This run uses Claude online, which costs money/)).toBeInTheDocument();
    expect(screen.getByText(/attribution with provider=anthropic is paid/)).toBeInTheDocument();
    expect(bodies).toHaveLength(1); // nothing auto-retries
    await user.click(screen.getByRole("button", { name: "confirm the paid run" }));
    await waitFor(() => expect(bodies).toHaveLength(2));
    expect(bodies[0]).toEqual({ chapters: [], provider: "anthropic", model: "claude-test" });
    expect(bodies[1]).toEqual({ chapters: [], provider: "anthropic", model: "claude-test", confirm_paid: true });
  });

  it("with Claude as the server default, 'this PC' reads with the LOCAL model and never says pull claude", async () => {
    const user = userEvent.setup();
    const server = routes({
      attribution: {
        provider: "anthropic",
        model: "claude-test",
        anthropic_model: "claude-test",
        prompt_version: "v6",
        hybrid: false,
      },
      ollama: { base_url: "x", reachable: true, model_installed: false },
    }).get("/api/settings", { kokoro_default_voice: "af_heart", attribution_provider: "anthropic", attribution_model: "qwen2.5:7b" });
    const bodies = recordAttribute(server, { paid: true });
    renderWithProviders(<AttributePanel book={fresh} />);

    // the Claude default isn't blocked by the local model's state
    await screen.findByText(/Claude online · claude-test/);
    expect(await startKey()).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "change" }));
    await user.click(screen.getByRole("button", { name: /speaker-finding provider/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: /this PC \(Ollama\)/ }));
    expect(await screen.findByRole("textbox", { name: "speaker-finding model" })).toHaveValue("qwen2.5:7b");
    expect(await screen.findByText("ollama pull qwen2.5:7b")).toBeInTheDocument();
    expect(screen.queryByText(/ollama pull claude/)).not.toBeInTheDocument();
    expect(await startKey()).toBeDisabled();
    expect(bodies).toHaveLength(0);
  });

  it("the start line is announced, and a job that shows up takes the focus to its strip", async () => {
    const user = userEvent.setup();
    const server = routes();
    recordAttribute(server);
    const { rerender } = renderWithProviders(<AttributePanel book={fresh} />);
    await screen.findByText(/qwen2.5:7b/);
    await user.click(await startKey());
    expect(await screen.findByRole("status")).toHaveTextContent(/Started\./);

    // the book's live job lands (the shelf refetch): the focused start button goes away
    rerender(<AttributePanel book={{ ...fresh, active_job: { job_id: "j", kind: "attribute", state: "queued" } }} />);
    const strip = await screen.findByText("finding the speakers");
    await waitFor(() => expect(document.activeElement).toBe(strip.closest(".drainstrip")));
  });

  it("a page that opens with the job already running doesn't steal the focus", async () => {
    routes();
    renderWithProviders(
      <AttributePanel book={{ ...fresh, active_job: { job_id: "j", kind: "attribute", state: "running" } }} />,
    );
    await screen.findByText("finding the speakers");
    expect(document.activeElement).toBe(document.body);
  });

  it("while the job runs it shows that instead of a start button", async () => {
    routes();
    renderWithProviders(
      <AttributePanel book={{ ...fresh, active_job: { job_id: "j", kind: "attribute", state: "running" } }} id="x" />,
    );
    expect(await screen.findByText("finding the speakers")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /find the speakers in/ })).not.toBeInTheDocument();
  });
});

describe("AttributePanel — re-run for chapters (Advanced)", () => {
  async function openAdvanced(user: ReturnType<typeof userEvent.setup>) {
    await user.click(await screen.findByText("Advanced: re-run speaker finding for chapters…"));
    return screen.findByRole("textbox", { name: "chapters to re-read" });
  }

  it("requires a different model than the one that found the speakers, and says why", async () => {
    const user = userEvent.setup();
    const server = routes().get("/api/books/demo/characters", overview());
    const bodies = recordAttribute(server);
    renderWithProviders(<AttributePanel book={found} chapterCount={10} />);

    const chapters = await openAdvanced(user);
    // the limits are said up front
    expect(screen.getByText(/same model gives the same result/)).toBeInTheDocument();
    expect(screen.getByText(/only runs when the whole book is read, so it is skipped here/)).toBeInTheDocument();
    expect(await screen.findByText(/found with: this PC \(Ollama\) · qwen2.5:7b/)).toBeInTheDocument();

    await user.type(chapters, "7, 3, 5-6");
    const key = screen.getByRole("button", { name: "re-run for 4 chapters" });
    expect(key).toBeDisabled();
    expect(key).toHaveAccessibleDescription(/found with qwen2.5:7b\. The same model gives the same answer/);

    const model = screen.getByRole("textbox", { name: "re-run model" });
    await user.clear(model);
    await user.type(model, "llama3.1:8b");
    expect(key).toBeEnabled();
    await user.click(key);
    await waitFor(() =>
      expect(bodies).toEqual([{ chapters: [3, 5, 6, 7], provider: "local", model: "llama3.1:8b" }]),
    );
    expect(await screen.findByText(/Started\./)).toBeInTheDocument();
  });

  it("allows the same model once Seiyuu's prompt has changed since the speakers were found", async () => {
    const user = userEvent.setup();
    // found with v5; the server now runs v6 — the cache key differs, so this is a real re-read
    const server = routes().get("/api/books/demo/characters", overview({ prompt_version: "v5" }));
    const bodies = recordAttribute(server);
    renderWithProviders(<AttributePanel book={found} chapterCount={10} />);

    await user.type(await openAdvanced(user), "5 - 7");
    expect(await screen.findByText(/instructions have been updated since these were found/)).toBeInTheDocument();
    const key = screen.getByRole("button", { name: "re-run for 3 chapters" });
    await waitFor(() => expect(key).toBeEnabled());
    await user.click(key);
    await waitFor(() => expect(bodies).toEqual([{ chapters: [5, 6, 7], provider: "local", model: "qwen2.5:7b" }]));
  });

  it("explains a bad chapter list instead of sending it", async () => {
    const user = userEvent.setup();
    routes().get("/api/books/demo/characters", overview({ model_id: "something-else" }));
    renderWithProviders(<AttributePanel book={found} chapterCount={4} />);
    const chapters = await openAdvanced(user);
    await screen.findByText(/found with: .* something-else/);
    await user.type(chapters, "2-9");
    const key = screen.getByRole("button", { name: "re-run for these chapters" });
    expect(key).toBeDisabled();
    expect(key).toHaveAccessibleDescription(/chapters 1 to 4/);
  });

  it("a Claude re-run keeps the paid confirm and sends the chapters with it", async () => {
    const user = userEvent.setup();
    const server = routes().get("/api/books/demo/characters", overview());
    const bodies = recordAttribute(server, { paid: true });
    renderWithProviders(<AttributePanel book={found} chapterCount={10} />);

    await user.type(await openAdvanced(user), "2");
    await user.click(screen.getByRole("button", { name: /re-run provider/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: /Claude online, paid/ }));
    await user.click(screen.getByRole("button", { name: "re-run for chapter 2" }));
    expect(await screen.findByText(/provider=anthropic is paid/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "confirm the paid run" }));
    await waitFor(() => expect(bodies).toHaveLength(2));
    expect(bodies[0]).toEqual({ chapters: [2], provider: "anthropic", model: "claude-test" });
    expect(bodies[1]).toEqual({ chapters: [2], provider: "anthropic", model: "claude-test", confirm_paid: true });
  });

  it("a local re-run is blocked by the probe like the first run", async () => {
    const user = userEvent.setup();
    routes({ ollama: { base_url: "x", reachable: false, model_installed: null } }).get(
      "/api/books/demo/characters",
      overview({ model_id: "older-model" }),
    );
    renderWithProviders(<AttributePanel book={found} chapterCount={10} />);
    await user.type(await openAdvanced(user), "1");
    await screen.findByText(/found with: .* older-model/);
    const key = screen.getByRole("button", { name: "re-run for chapter 1" });
    expect(key).toBeDisabled();
    expect(key).toHaveAccessibleDescription(/Ollama isn't running/);
    expect(screen.getByText("ollama serve")).toBeInTheDocument();
  });
});
