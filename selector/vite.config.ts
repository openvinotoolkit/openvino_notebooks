import react from '@vitejs/plugin-react';
import { join, resolve } from 'path';
import { URL } from 'url';
import { defineConfig, loadEnv } from 'vite';
import svgr from 'vite-plugin-svgr';

import { generateNotebooksMapFilePlugin } from './src/notebook-metadata/generate-notebooks-map.js';

const ENV_DIR = join(__dirname, 'vite-env');

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const viteEnv = loadEnv(mode, ENV_DIR);

  return {
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
    build: {
      outDir: join('dist', 'openvino_notebooks'),
      modulePreload: {
        polyfill: false,
      },
      rollupOptions: {
        input: {
          index: resolve(__dirname, 'index.html'),
          embedded: resolve(__dirname, 'embedded.html'),
        },
        output: {
          entryFileNames({ name, isEntry }) {
            if (name === 'embedded' && isEntry) {
              return 'assets/[name].js';
            }
            return 'assets/[name]-[hash].js';
          },
        },
      },
    },
    envDir: ENV_DIR,
    experimental: {
      renderBuiltUrl(filename: string, { hostId }: { hostId: string }) {
        if (hostId === 'embedded.html') {
          return new URL(filename, viteEnv.VITE_APP_LOCATION).toString();
        }
      },
    },
  };
});
