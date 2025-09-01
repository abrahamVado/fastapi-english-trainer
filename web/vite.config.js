// web/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "fs";
import path from "path";

const certDir = path.resolve(__dirname, "../certs"); // <-- parent folder

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,               // expose on LAN
    port: 5173,
    https: {
      key:  fs.readFileSync(path.join(certDir, "server.key")),
      cert: fs.readFileSync(path.join(certDir, "server.crt")), // <-- .crt
    },
    origin: "https://trainer.local:5173",
    hmr: {
      host: "trainer.local",
      protocol: "wss",
      port: 5173,
      // If still flaky behind a proxy/VPN, try:
      // clientPort: 5173,
      // overlay: true,
    },
  },
});
