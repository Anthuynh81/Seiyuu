import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev-time: the FastAPI server (uvicorn seiyuu.api.main:app) owns /api; proxying keeps
// the app origin-clean so CORS never enters the picture.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: {
    environment: "jsdom", // buildClipWords creates real spans
    include: ["src/**/*.test.ts"],
  },
});
