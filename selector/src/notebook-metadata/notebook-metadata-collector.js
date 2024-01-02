// @ts-check

import { execSync } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { basename, dirname, join } from 'path';
import { fileURLToPath } from 'url';

/** @typedef {import('../models/notebook-metadata.ts').INotebookMetadata} INotebookMetadata */
/**
 * @typedef {{
 *  metadata: { openvino_notebooks?: Partial<INotebookMetadata> };
 *  cells: Array<{ cell_type: 'markdown' | 'code'; source: string[]; }>
 * }} INotebookJson
 */

const CURRENT_DIR_PATH = dirname(fileURLToPath(import.meta.url));

export const NOTEBOOKS_DIRECTORY_PATH = join(CURRENT_DIR_PATH, '..', '..', '..', 'notebooks');

export class NotebookMetadataCollector {
  /**
   * @param {string} notebookFilePath
   */
  constructor(notebookFilePath) {
    /** @private */
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
   * @private
   * @returns {string}
   */
  get _absoluteNotebookPath() {
    return join(NOTEBOOKS_DIRECTORY_PATH, this._notebookFilePath);
  }

  /**
   * @private
   * @returns {string}
   */
  get _notebookFileName() {
    return basename(this._notebookFilePath);
  }

  /**
   * @private
   * @returns {INotebookJson}
   */
  _getNotebookJson() {
    const notebookContent = readFileSync(this._absoluteNotebookPath, { encoding: 'utf8' });
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return JSON.parse(notebookContent);
  }

  /**
   * @private
   * @template {keyof INotebookMetadata} K
   * @param {K} key
   * @returns {Partial<INotebookMetadata>[K] | null}
   */
  _getNotebookFileMetadata(key) {
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

  /**
   * @private
   * @returns {string}
   */
  _getReadmeContent() {
    return readFileSync(this._readmeFilePath, { encoding: 'utf8' });
  }

  /**
   * @private
   * @returns {string}
   */
  _getNotebookTitle() {
    const { cells } = this._getNotebookJson();
    const firstCellContent = cells[0].source.join('');
    const titleRegexp = /# (?<title>.+)/g;
    const match = titleRegexp.exec(firstCellContent);
    if (!match || !match.groups || !match.groups.title) {
      return '';
    }
    return match.groups.title;
  }

  /**
   * @private
   * @returns {string}
   */
  _getNotebookDescription() {
    const description = this._getNotebookFileMetadata('description');
    return description || '';
  }

  /**
   * @private
   * @returns {string | null}
   */
  _getImageUrl() {
    const imageUrl = this._getNotebookFileMetadata('imageUrl');
    return imageUrl || null;
  }

  /**
   * @private
   * @returns {string}
   */
  _getNotebookCreatedDate() {
    return execSync(
      `git log --pretty=format:"%ad" --date=iso --diff-filter=A -- ${this._absoluteNotebookPath}`
    ).toString();
  }

  /**
   * @private
   * @returns {string}
   */
  _getNotebookModifiedDate() {
    return execSync(
      `git log -1 --pretty=format:"%cd" --date=iso --diff-filter=M -- ${this._absoluteNotebookPath}`
    ).toString();
  }

  /**
   * @private
   * @returns {string}
   */
  _getNotebookGitHubLink() {
    return `https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/${this._notebookFilePath}`;
  }

  /**
   * @private
   * @returns {string | null}
   */
  _getNotebookColabLink() {
    const readmeContent = this._getReadmeContent();
    const colabBadgeRegExp = new RegExp(
      `\\[!\\[Colab\\]\\(.+\\)\\]\\((?<link>.+(?:${this._notebookFileName}))\\)`,
      'g'
    );
    const match = colabBadgeRegExp.exec(readmeContent);
    if (!match || !match.groups || !match.groups.link) {
      return null;
    }
    return match.groups.link;
  }

  /**
   * @private
   * @returns {string | null}
   */
  _getNotebookBinderLink() {
    const readmeContent = this._getReadmeContent();
    const binderBadgeRegExp = new RegExp(
      `\\[!\\[Binder\\]\\(.+\\)\\]\\((?<link>.+(?:${this._notebookFileName}))\\)`,
      'g'
    );
    const match = binderBadgeRegExp.exec(readmeContent);
    if (!match || !match.groups || !match.groups.link) {
      return null;
    }
    return match.groups.link;
  }

  /**
   * @private
   * @returns {INotebookMetadata['tags']}
   */
  _getTags() {
    // TODO Consider merging of tags keys
    const tags = this._getNotebookFileMetadata('tags');
    return (
      tags || {
        categories: [],
        tasks: [],
        libraries: [],
        other: [],
      }
    );
  }

  /**
   * Collects and returns new metadata object
   *
   * @public
   * @returns {INotebookMetadata}
   */
  getMetadata() {
    return {
      title: this._getNotebookTitle(),
      description: this._getNotebookDescription(),
      path: this._notebookFilePath,
      imageUrl: this._getImageUrl(),
      createdDate: this._getNotebookCreatedDate(),
      modifiedDate: this._getNotebookModifiedDate(),
      links: {
        github: this._getNotebookGitHubLink(),
        colab: this._getNotebookColabLink(),
        binder: this._getNotebookBinderLink(),
      },
      tags: this._getTags(),
    };
  }
}
