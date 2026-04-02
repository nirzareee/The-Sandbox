import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: '../frontend',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/ws': {
        target: 'http://localhost:8000',
        ws: true,
      },
      '/start': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/state': 'http://localhost:8000',
      '/agents': 'http://localhost:8000',
      '/setup': 'http://localhost:8000',
      '/debug': 'http://localhost:8000',
    },
  },
})
