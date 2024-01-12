// @ts-check

import { existsSync, readFileSync, writeFileSync } from 'fs';
import { join, resolve } from 'path';

import { NotebookMetadataHandler } from './notebook-metadata-handler.js';

export const NOTEBOOKS_MAP_FILE_NAME = 'notebooks-metadata-map.json';

/**
 *
 * @param {string} path
 * @returns {void}
 */
function generateNotebooksMapFile(path) {
  /** @typedef {import("./notebook-metadata-collector.js").INotebookMetadata} INotebookMetadata */

  /** @type {Record<string, INotebookMetadata>} */
  const notebooksMetadataMap = {};

  const notebooksPaths = NotebookMetadataHandler.getNotebooksPaths();

  for (const notebookPath of notebooksPaths) {
    console.info(`Collecting metadata for notebook "${notebookPath}"`);
    const { metadata } = new NotebookMetadataHandler(notebookPath);
    notebooksMetadataMap[notebookPath] = metadata;
  }

  writeFileSync(join(path, NOTEBOOKS_MAP_FILE_NAME), JSON.stringify(notebooksMetadataMap, null, 2), {
    flag: 'w',
  });
}

/**
 *
 * @returns {import('vite').PluginOption}
 */
export const generateNotebooksMapFilePlugin = () => {
  /** @type {import('vite').ResolvedConfig} */
  let config;

  let distPath = '';

  return {
    name: 'generate-notebooks-map-file',
    configResolved(resolvedConfig) {
      config = resolvedConfig;
      distPath = resolve(config.root, config.build.outDir);
    },
    buildStart() {
      const notebooksMapFileExists = existsSync(join(distPath, NOTEBOOKS_MAP_FILE_NAME));
      if (config.command === 'serve' && notebooksMapFileExists) {
        console.info(
          `"${NOTEBOOKS_MAP_FILE_NAME}" file already exists and is served from "${distPath}" dist directory.`
        );
        return;
      }
      console.info(`Creating notebooks map file...`);
      generateNotebooksMapFile(distPath);
      console.info(`Notebooks map file is created in "${distPath}".`);
    },
    configureServer(devServer) {
      devServer.middlewares.use(`/${NOTEBOOKS_MAP_FILE_NAME}`, (_, res) => {
        const notebooksFileMapContent = readFileSync(join(distPath, NOTEBOOKS_MAP_FILE_NAME), { encoding: 'utf8' });
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.write(notebooksFileMapContent);
        res.end();
      });
    },
  };
};
