import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/redact": "http://localhost:8000",
      "/download": "http://localhost:8000",
      "/healthz": "http://localhost:8000",
    },
  },
});
