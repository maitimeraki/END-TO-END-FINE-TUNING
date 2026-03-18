import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy API calls during development to the local backend
      "/v1": "http://127.0.0.1:8000",
    },
  },
});
