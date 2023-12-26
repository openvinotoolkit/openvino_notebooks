// @ts-check

import { globSync } from 'glob';

import { NotebookMetadataCollector, NOTEBOOKS_DIRECTORY_PATH } from './notebook-metadata-collector.js';
import { toMarkdown } from './notebook-metadata-formatter.js';
import { NotebookMetadataValidationError, validateNotebookMetadata } from './notebook-metadata-validator.js';

/** @typedef {import('../models/notebook').INotebookMetadata} INotebookMetadata */

export class NotebookMetadataHandler {
  /**
   * @param {string} notebookFilePath
   */
  constructor(notebookFilePath) {
    /** @private */
    this._notebookFilePath = notebookFilePath;

    /** @private */
    this._metadata = new NotebookMetadataCollector(this._notebookFilePath).getMetadata();
  }

  /**
   * @throws {NotebookMetadataValidationError} - All invalid metadata properties
   */
  validateMetadata() {
    validateNotebookMetadata(this._metadata);
  }

  toMarkdown() {
    return toMarkdown(this._metadata);
  }

  static getNotebooksPaths() {
    // TODO Consider removing external glob dependency
    return globSync('**/*.ipynb', {
      ignore: ['**/.ipynb_checkpoints/*', '**/notebook_utils.ipynb'],
      cwd: NOTEBOOKS_DIRECTORY_PATH,
    });
  }

  /**
   * @param {string[]} notebooksPaths
   * @throws {NotebookMetadataValidationError} - All invalid metadata properties for each notebook
   */
  static validateNotebooks(notebooksPaths) {
    for (const notebookPath of notebooksPaths) {
      const errors = [];
      try {
        new NotebookMetadataHandler(notebookPath).validateMetadata();
      } catch (error) {
        if (error instanceof NotebookMetadataValidationError) {
          errors.push(error.message);
        } else {
          throw error;
        }
      }
      if (errors.length) {
        throw new NotebookMetadataValidationError(errors.join('\n\n'));
      }
    }
  }

  /**
   * @throws {NotebookMetadataValidationError} - All invalid metadata properties for each notebook
   */
  static validateAll() {
    const notebooksPaths = NotebookMetadataHandler.getNotebooksPaths();
    NotebookMetadataHandler.validateNotebooks(notebooksPaths);
  }
}
