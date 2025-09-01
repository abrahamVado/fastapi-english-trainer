// web/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "fs";
import path from "path";

// Use the mkcert files you created for trainer.local
// Example if you placed them in ../certs:
//   trainer.local+1.pem          (cert)
//   trainer.local+1-key.pem      (key)
const certDir = path.resolve(__dirname, "../cerd");

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,           // LAN access
    port: 5173,
    https: {
      key:  fs.readFileSync(path.join(certDir, "localhost+1-key.pem")),
      cert: fs.readFileSync(path.join(certDir, "localhost+1.pem")),
    },
    origin: "https://trainer.local:5173",
    hmr: {
      host: "trainer.local",
      protocol: "wss",
      port: 5173,
    },
    // 🚀 Proxy API to backend to avoid CORS
    proxy: {
      "/api": {
        target: "https://trainer.local:8000",
        changeOrigin: true,
        secure: false, // backend uses mkcert/self-signed
      },
      "/health": {
        target: "https://trainer.local:8000",
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
