import { defineConfig } from 'vite'
import fs from 'fs'
import svgr from 'vite-plugin-svgr'
import react from '@vitejs/plugin-react-swc'


// https://vite.dev/config/
export default defineConfig({
  base: '/dom/',
  plugins: [react(), svgr()],
  server: {
    https: {
      key: fs.readFileSync('./server.key'),
      cert: fs.readFileSync('./server.crt'),
    },
    host: '0.0.0.0',      // Listen on all addresses, including external
    port: 5400     // You can choose a different port if needed
  },
})