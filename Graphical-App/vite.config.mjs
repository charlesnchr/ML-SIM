import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The renderer is loaded from file:// in the built app, so asset paths are relative.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "build",
    // Blueprint 3 ships selectors that the CSS minifier rejects.
    cssMinify: false,
    chunkSizeWarningLimit: 2000,
  },
  server: {
    port: 4001,
    strictPort: true,
  },
});
