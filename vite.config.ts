import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // React references process.env.NODE_ENV; Vite's lib mode doesn't replace it.
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    lib: {
      entry: "src/index.tsx",
      name: "GemmaVLLMMetrics",
      // Forces .js instead of Vite's default .umd.cjs to match fiftyone.script.
      fileName: () => "index.umd.js",
      formats: ["umd"],
    },
    rollupOptions: {
      external: ["react", "react-dom", "@fiftyone/operators", "@fiftyone/plugins"],
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
          "@fiftyone/operators": "__foo__",
          "@fiftyone/plugins": "__fop__",
        },
      },
    },
  },
});
