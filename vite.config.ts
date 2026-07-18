import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Vite's lib-mode build doesn't replace process.env.NODE_ENV the way its
  // app-mode build does, but React and other deps still reference it --
  // without this, the bundle throws "process is not defined" in the browser.
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    lib: {
      entry: "src/index.tsx",
      name: "GemmaVLLMMetrics",
      // Force the .js extension explicitly -- with "type": "module" in
      // package.json, Vite's default naming emits UMD output as .umd.cjs,
      // which wouldn't match the "fiftyone.script" path FiftyOne's App
      // loader expects.
      fileName: () => "index.umd.js",
      formats: ["umd"],
    },
    rollupOptions: {
      external: [
        "react",
        "react-dom",
        "@fiftyone/components",
        "@fiftyone/operators",
        "@fiftyone/plugins",
        "@fiftyone/state",
        "recoil",
      ],
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
          recoil: "recoil",
          "@fiftyone/components": "__foc__",
          "@fiftyone/operators": "__foo__",
          "@fiftyone/plugins": "__fop__",
          "@fiftyone/state": "__fos__",
        },
      },
    },
  },
});
