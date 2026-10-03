import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In dev, /api/* is proxied to the serving API (override with VITE_API_URL). In production nginx does the same.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { '/api': process.env.VITE_API_URL || 'http://localhost:8000' },
  },
})
