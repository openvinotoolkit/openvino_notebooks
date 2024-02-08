import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { defineConfig } from 'vite';
import svgr from 'vite-plugin-svgr';

import { generateNotebooksMapFilePlugin } from './src/notebook-metadata/generate-notebooks-map.js';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), svgr(), generateNotebooksMapFilePlugin()],
  base: '/openvino_notebooks/',
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@assets': resolve(__dirname, './src/assets'),
      '@components': resolve(__dirname, './src/components'),
      '@spark-design': resolve(__dirname, './src/@spark-design'),
    },
  },
});
