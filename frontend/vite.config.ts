import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev-time: the FastAPI server (uvicorn seiyuu.api.main:app) owns /api; proxying keeps
// the app origin-clean so CORS never enters the picture.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
