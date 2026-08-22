import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Bind IPv4 explicitly. Left to itself Vite listens on ::1 only on this
    // machine, and a browser that resolves "localhost" to 127.0.0.1 first gets
    // connection refused with no useful message. Found by pointing Chrome at
    // the dev server and getting an error page while curl on ::1 answered 200 --
    // not something to discover in front of a panel.
    host: "127.0.0.1",
    // Proxy keeps the frontend origin-clean in dev: fetch("/api/analyze")
    // hits FastAPI on :8000 without CORS games.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
