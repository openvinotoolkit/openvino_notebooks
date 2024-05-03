// @ts-check

import { existsSync, readFileSync } from 'fs';
import { join, resolve } from 'path';

import { generateNotebooksMetadataFile } from '../notebook-metadata/generate-notebooks-map.js';
import { createBuildChecksumFile } from './build-checksum.js';
import { NOTEBOOKS_METADATA_FILE_NAME, NOTEBOOKS_STATUS_FILE_NAME } from './constants.js';
import { fetchNotebooksStatusFile } from './fetch-notebooks-status.js';

/**
 *
 * @returns {import('vite').PluginOption}
 */
export const collectNotebooksFilesPlugin = () => {
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
        await generateNotebooksMetadataFile(distPath);
        await fetchNotebooksStatusFile(distPath);
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
        await generateNotebooksMetadataFile(distPath);
      }
      const notebooksStatusFileExists = existsSync(join(distPath, NOTEBOOKS_STATUS_FILE_NAME));
      if (notebooksStatusFileExists) {
        console.info(
          `"${NOTEBOOKS_STATUS_FILE_NAME}" file already exists and is served from "${distPath}" dist directory.`
        );
      } else {
        // TODO Consider generating mock file
        await fetchNotebooksStatusFile(distPath);
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
