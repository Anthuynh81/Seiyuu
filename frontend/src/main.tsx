import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import App from "./App";
import { PlayerProvider } from "./app/player";

// Talkback v4 type trio, self-hosted (no CDN at runtime): Plex Mono is the machine
// talking, Plex Sans is the interface, Literata is the book on paper.
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/ibm-plex-mono/700.css";
import "@fontsource-variable/ibm-plex-sans/index.css";
import "@fontsource-variable/literata/index.css";
import "@fontsource-variable/literata/wght-italic.css";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      retry: 1, // API errors are typed and actionable; hammering them helps nobody
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <PlayerProvider>
          <App />
        </PlayerProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
