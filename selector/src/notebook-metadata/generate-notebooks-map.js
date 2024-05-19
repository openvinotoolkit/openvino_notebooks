// @ts-check

import { existsSync, mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';

import { NOTEBOOKS_METADATA_FILE_NAME } from '../shared/constants.js';
import { NotebookMetadataValidationError } from './notebook-metadata-validator.js';

/**
 *
 * @param {string} path
 * @throws {NotebookMetadataValidationError}
 * @returns {Promise<void>}
 */
export async function generateNotebooksMetadataFile(path) {
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
