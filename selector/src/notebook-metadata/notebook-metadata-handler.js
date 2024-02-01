// @ts-check

import { globSync } from 'glob';

import { NotebookMetadataCollector, NOTEBOOKS_DIRECTORY_PATH } from './notebook-metadata-collector.js';
import { toMarkdown } from './notebook-metadata-formatter.js';
import { NotebookMetadataValidationError, validateNotebookMetadata } from './notebook-metadata-validator.js';

/** @typedef {import('../models/notebook-metadata.ts').INotebookMetadata} INotebookMetadata */

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

  /** @private @const */
  static _skippedNotebooks = [
    '110-ct-segmentation-quantize/data-preparation-ct-scan.ipynb',
    '110-ct-segmentation-quantize/pytorch-monai-training.ipynb',
  ];

  /**
   * @returns {string | null} Validation error
   */
  validateMetadata() {
    try {
      validateNotebookMetadata(this._metadata);
    } catch (error) {
      if (error instanceof NotebookMetadataValidationError) {
        error.message = `Invalid metadata for notebook "${this._notebookFilePath}".\n${error.message}`;
        return error.message;
      }
      throw error;
    }
    return null;
  }

  /**
   * @param {boolean} hasError
   */
  toMarkdown(hasError) {
    return toMarkdown(this._metadata, hasError);
  }

  static getNotebooksPaths() {
    // TODO Consider removing external glob dependency
    return globSync('**/*.ipynb', {
      ignore: ['**/.ipynb_checkpoints/*', '**/notebook_utils.ipynb', ...this._skippedNotebooks],
      cwd: NOTEBOOKS_DIRECTORY_PATH,
    });
  }

  /**
   * @param {string[]} notebooksPaths
   * @return {[string | null, string[]]} Tuple of validation error (if exist) and array of metadata markdown representations
   */
  static validateNotebooks(notebooksPaths) {
    const errors = [];
    const metadataMarkdowns = [];
    for (const notebookPath of notebooksPaths) {
      if (NotebookMetadataHandler._skippedNotebooks.includes(notebookPath)) {
        continue;
      }
      const notebookMetadataHandler = new NotebookMetadataHandler(notebookPath);
      const error = notebookMetadataHandler.validateMetadata();
      if (error) {
        errors.push(error);
      }
      const metadataMarkdown = notebookMetadataHandler.toMarkdown(Boolean(error));
      metadataMarkdowns.push(metadataMarkdown);
    }
    return [errors.length ? errors.join('\n\n') : null, metadataMarkdowns];
  }

  static validateAll() {
    const notebooksPaths = NotebookMetadataHandler.getNotebooksPaths();
    return NotebookMetadataHandler.validateNotebooks(notebooksPaths);
  }
}
