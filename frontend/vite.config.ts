import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    react(),
    mode === 'development' &&
    componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  css: {
    // Ensure consistent CSS processing between dev and prod
    devSourcemap: true,
    postcss: {
      plugins: []
    }
  },
  build: {
    // Ensure CSS is processed the same way in production
    cssCodeSplit: false,
    minify: 'esbuild',
    sourcemap: true
  },
}));
