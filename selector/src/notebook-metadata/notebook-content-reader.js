// @ts-check

import { existsSync, readFileSync } from 'fs';
import { basename, dirname, join } from 'path';
import { fileURLToPath } from 'url';

/** @typedef {import('../shared/notebook-metadata.ts').INotebookMetadata} INotebookMetadata */
/**
 * @typedef {{
 *  metadata: { openvino_notebooks?: Partial<INotebookMetadata> };
 *  cells: Array<{ cell_type: 'markdown' | 'code'; source: string[]; }>
 * }} INotebookJson
 */

const CURRENT_DIR_PATH = dirname(fileURLToPath(import.meta.url));

export const NOTEBOOKS_DIRECTORY_PATH = join(CURRENT_DIR_PATH, '..', '..', '..', 'notebooks');

export class NotebookContentReader {
  /**
   * @param {string} notebookFilePath
   */
  constructor(notebookFilePath) {
    /** @protected */
    this._notebookFilePath = notebookFilePath;

    this._checkFilesExist();
  }

  /**
   * @private
   */
  _checkFilesExist() {
    if (!existsSync(this._absoluteNotebookPath)) {
      throw Error(`Notebook file "${this._notebookFilePath}" does not exists.`);
    }

    if (!existsSync(this._readmeFilePath)) {
      throw Error(`README.md file does not exists for notebook "${this._notebookFilePath}".`);
    }
  }

  /**
   * @private
   * @returns {string}
   */
  get _readmeFilePath() {
    return join(NOTEBOOKS_DIRECTORY_PATH, dirname(this._notebookFilePath), 'README.md');
  }

  /**
   * @protected
   * @returns {string}
   */
  get _absoluteNotebookPath() {
    return join(NOTEBOOKS_DIRECTORY_PATH, this._notebookFilePath);
  }

  /**
   * @protected
   * @returns {string}
   */
  get _notebookFileName() {
    return basename(this._notebookFilePath);
  }

  /**
   * @protected
   * @returns {INotebookJson}
   */
  _getNotebookJson() {
    const notebookContent = readFileSync(this._absoluteNotebookPath, { encoding: 'utf8' });
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return JSON.parse(notebookContent);
  }

  /**
   * @protected
   * @returns {INotebookJson['cells']}
   */
  _getCodeCells() {
    return this._getNotebookJson().cells.filter(({ cell_type }) => cell_type === 'code');
  }

  /**
   * @protected
   * @returns {string}
   */
  _getReadmeContent() {
    return readFileSync(this._readmeFilePath, { encoding: 'utf8' });
  }

  /**
   * @protected
   * @template {keyof INotebookMetadata} K
   * @param {K} key
   * @returns {Partial<INotebookMetadata>[K] | null}
   */
  _getMetadataFromNotebookFile(key) {
    const { metadata } = this._getNotebookJson();
    if (!metadata.openvino_notebooks) {
      console.warn(`No "openvino_notebooks" metadata found in notebook "${this._notebookFilePath}".`);
      return null;
    }
    const metadataPart = metadata.openvino_notebooks[key];
    if (metadataPart === undefined) {
      console.warn(`"${key}" is not found in "openvino_notebooks" metadata for notebook "${this._notebookFilePath}".`);
      return null;
    }
    return metadataPart;
  }
}
