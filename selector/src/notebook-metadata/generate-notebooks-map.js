// @ts-check

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join, resolve } from 'path';

import { createBuildChecksumFile } from '../shared/build-checksum.js';
import { NotebookMetadataValidationError } from './notebook-metadata-validator.js';

export const NOTEBOOKS_MAP_FILE_NAME = 'notebooks-metadata-map.json';

/**
 *
 * @param {string} path
 * @throws {NotebookMetadataValidationError}
 * @returns {Promise<void>}
 */
async function generateNotebooksMapFile(path) {
  /** @typedef {import("./notebook-metadata-collector.js").INotebookMetadata} INotebookMetadata */

  /** @type {Record<string, INotebookMetadata>} */
  const notebooksMetadataMap = {};

  console.info(`Creating notebooks map file...`);

  const { NotebookMetadataHandler } = await import('./notebook-metadata-handler.js');

  const notebooksPaths = NotebookMetadataHandler.getNotebooksPaths();

  for (const notebookPath of notebooksPaths) {
    console.info(`Collecting metadata for notebook "${notebookPath}"`);
    const notebookMetadataHandler = new NotebookMetadataHandler(notebookPath);
    const error = notebookMetadataHandler.validateMetadata();
    if (error) {
      throw new NotebookMetadataValidationError(error);
    }
    notebooksMetadataMap[notebookPath] = notebookMetadataHandler.metadata;
  }

  if (!existsSync(path)) {
    mkdirSync(path, { recursive: true });
  }

  writeFileSync(join(path, NOTEBOOKS_MAP_FILE_NAME), JSON.stringify(notebooksMetadataMap, null, 2), {
    flag: 'w',
  });

  console.info(`Notebooks map file is created in "${path}".`);
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
    async closeBundle() {
      if (config.command === 'build') {
        await generateNotebooksMapFile(distPath);
        await createBuildChecksumFile(distPath);
      }
    },
    async configureServer(devServer) {
      const notebooksMapFileExists = existsSync(join(distPath, NOTEBOOKS_MAP_FILE_NAME));
      if (notebooksMapFileExists) {
        console.info(
          `"${NOTEBOOKS_MAP_FILE_NAME}" file already exists and is served from "${distPath}" dist directory.`
        );
      } else {
        await generateNotebooksMapFile(distPath);
      }

      devServer.middlewares.use(`${config.base}${NOTEBOOKS_MAP_FILE_NAME}`, (_, res) => {
        const notebooksFileMapContent = readFileSync(join(distPath, NOTEBOOKS_MAP_FILE_NAME), { encoding: 'utf8' });
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.write(notebooksFileMapContent);
        res.end();
      });
    },
  };
};
