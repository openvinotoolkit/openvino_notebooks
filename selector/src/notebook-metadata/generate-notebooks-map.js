// @ts-check

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join, resolve } from 'path';

import { createBuildChecksumFile } from '../shared/build-checksum.js';
import { fetchNotebooksStatusesFile, NOTEBOOKS_STATUS_FILE_NAME } from '../shared/fetch-notebooks-status.js';
import { NotebookMetadataValidationError } from './notebook-metadata-validator.js';

export const NOTEBOOKS_METADATA_FILE_NAME = 'notebooks-metadata-map.json';

/**
 *
 * @param {string} path
 * @throws {NotebookMetadataValidationError}
 * @returns {Promise<void>}
 */
// TODO Consider renaming to `generateNotebooksMetadataFile`
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

  writeFileSync(join(path, NOTEBOOKS_METADATA_FILE_NAME), JSON.stringify(notebooksMetadataMap, null, 2), {
    flag: 'w',
  });

  console.info(`Notebooks map file is created in "${path}".`);
}

/**
 *
 * @returns {import('vite').PluginOption}
 */
// TODO Consider renaming (e.g. collectNotebooksFiles)
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
        await fetchNotebooksStatusesFile(distPath);
        await createBuildChecksumFile(distPath);
      }
    },
    async configureServer(devServer) {
      const notebooksMapFileExists = existsSync(join(distPath, NOTEBOOKS_METADATA_FILE_NAME));
      if (notebooksMapFileExists) {
        console.info(
          `"${NOTEBOOKS_METADATA_FILE_NAME}" file already exists and is served from "${distPath}" dist directory.`
        );
      } else {
        await generateNotebooksMapFile(distPath);
      }
      const notebooksStatusFileExists = existsSync(join(distPath, NOTEBOOKS_STATUS_FILE_NAME));
      if (notebooksStatusFileExists) {
        console.info(
          `"${NOTEBOOKS_STATUS_FILE_NAME}" file already exists and is served from "${distPath}" dist directory.`
        );
      } else {
        // TODO Consider generating mock file
        await fetchNotebooksStatusesFile(distPath);
      }

      devServer.middlewares.use(...getFileMiddleware(NOTEBOOKS_METADATA_FILE_NAME, config.base, distPath));
      devServer.middlewares.use(...getFileMiddleware(NOTEBOOKS_STATUS_FILE_NAME, config.base, distPath));
    },
  };
};

/**
 * @param {string} fileName
 * @param {string} urlBase
 * @param {string} distPath
 * @returns {[string, import('vite').Connect.NextHandleFunction]}
 */
function getFileMiddleware(fileName, urlBase, distPath) {
  const route = `${urlBase}${fileName}`;
  /** @type {import('vite').Connect.NextHandleFunction} */
  const handler = (_, res) => {
    const fileContent = readFileSync(join(distPath, fileName), {
      encoding: 'utf8',
    });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.write(fileContent);
    res.end();
  };
  return [route, handler];
}
