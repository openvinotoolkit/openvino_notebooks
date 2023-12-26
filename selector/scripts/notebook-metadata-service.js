// @ts-check

import { execSync } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { globSync } from 'glob';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

import { validateNotebookMetadata } from '../src/notebook-metadata/notebook-metadata-validator.js';

/**
 * @typedef {import('../src/models/notebook').INotebookMetadata} INotebookMetadata
 */

const SCRIPT_DIR_PATH = dirname(fileURLToPath(import.meta.url));

const NOTEBOOKS_DIRECTORY_PATH = join(SCRIPT_DIR_PATH, '..', '..', 'notebooks');

/**
 * Returns path to README.md file releated to the notebook path
 *
 * @param {string} notebookPath
 * @returns {string}
 */
function getNotebookReadmeFilePath(notebookPath) {
  return join(NOTEBOOKS_DIRECTORY_PATH, dirname(notebookPath), 'README.md');
}

export class NotebookMetadataService {
  /**
   * @param {string} notebookFilePath
   */
  constructor(notebookFilePath) {
    /** @private */
    this._notebookFilePath = notebookFilePath;
    /** @private */
    this._readmeFilePath = getNotebookReadmeFilePath(this._notebookFilePath);

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
   * Returns absolute notebook wile path
   *
   * @private
   * @returns {string}
   */
  get _absoluteNotebookPath() {
    return join(NOTEBOOKS_DIRECTORY_PATH, this._notebookFilePath);
  }

  /**
   * Parses title from README.md file related to notebook
   *
   * @private
   * @returns {string | null}
   */
  _getTitleFromReadme() {
    const titleRegexp = /# (?<title>.+)/g;
    const readmeContent = readFileSync(this._readmeFilePath, { encoding: 'utf8' });
    const match = titleRegexp.exec(readmeContent);
    if (!match || !match.groups || !match.groups.title) {
      return null;
    }
    return match.groups.title;
  }

  /**
   * Parses title from README.md file related to notebook
   *
   * @private
   * @returns {string | null}
   */
  _getTitleFromNotebook() {
    const titleRegexp = /# (?<title>.+)/g;
    const notebookContent = readFileSync(this._absoluteNotebookPath, { encoding: 'utf8' });
    const { cells } = JSON.parse(notebookContent);
    const firstCellContent = cells[0].source.join('');
    const match = titleRegexp.exec(firstCellContent);
    if (!match || !match.groups || !match.groups.title) {
      return null;
    }
    return match.groups.title;
  }

  /**
   * Returns GitHub link to corresponding notebook
   *
   * @private
   * @returns {string}
   */
  _getNotebookGitHubLink() {
    return `https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/${this._notebookFilePath}`;
  }

  /**
   * Returns Colab link to corresponding notebook
   *
   * @private
   * @returns {string | null}
   */
  _getNotebookColabLink() {
    const colabBadgeRegExp = /\[!\[Colab\]\(.+\)\]\((?<link>.+)\)/g;
    const readmeContent = readFileSync(this._readmeFilePath, { encoding: 'utf8' });
    const match = colabBadgeRegExp.exec(readmeContent);
    if (!match || !match.groups || !match.groups.link) {
      return null;
    }
    return match.groups.link;
  }

  /**
   * Returns Binder link to corresponding notebook
   *
   * @private
   * @returns {string | null}
   */
  _getNotebookBinderLink() {
    const colabBadgeRegExp = /\[!\[Binder\]\(.+\)\]\((?<link>.+)\)/g;
    const readmeContent = readFileSync(this._readmeFilePath, { encoding: 'utf8' });
    const match = colabBadgeRegExp.exec(readmeContent);
    if (!match || !match.groups || !match.groups.link) {
      return null;
    }
    return match.groups.link;
  }

  /**
   * Returns notebook file created date
   *
   * @private
   * @returns {string}
   */
  _getNotebookCreatedDate() {
    return execSync(
      `git log --pretty=format:"%ad" --date=iso --diff-filter=A -- ${this._absoluteNotebookPath}`
    ).toString();
  }

  /**
   * Returns notebook file modified date
   *
   * @private
   * @returns {string}
   */
  _getNotebookModifiedDate() {
    return execSync(
      `git log -1 --pretty=format:"%cd" --date=iso --diff-filter=M -- ${this._absoluteNotebookPath}`
    ).toString();
  }

  /**
   * Generates new metadata object
   *
   * @public
   * @returns {INotebookMetadata}
   */
  _generateMetadata() {
    // TODO Move parser functions to separate module
    return {
      title: this._getTitleFromNotebook() || '',
      description: '', // TODO Add description parser
      path: this._notebookFilePath,
      imageUrl: null, // TODO Add image url parser
      createdDate: this._getNotebookCreatedDate(),
      modifiedDate: this._getNotebookModifiedDate(),
      links: {
        github: this._getNotebookGitHubLink(),
        colab: this._getNotebookColabLink(),
        binder: this._getNotebookBinderLink(),
      },
      tags: {
        // TODO Add tags parser
        categories: [],
        tasks: [],
        libraries: [],
        other: [],
      },
    };
  }

  /**
   * Validates metadata file for corresponding notebook
   *
   * @returns {void}
   */
  validateMetadataFile() {
    console.info(`Validating metadata for notebook "${this._notebookFilePath}"...`);
    const metadata = this._generateMetadata();
    try {
      validateNotebookMetadata(metadata);
    } catch (error) {
      throw Error(`Invalid metadata for notebook "${this._notebookFilePath}".\n${error}`);
    }
    console.info(`Metadata is valid.`);
  }

  static getNotebooksPaths() {
    return globSync('**/*.ipynb', {
      ignore: ['**/.ipynb_checkpoints/*', '**/notebook_utils.ipynb'],
      cwd: NOTEBOOKS_DIRECTORY_PATH,
    });
  }

  /**
   * Returns map with metadata objects for all notebooks
   *
   * @returns {Record<string, INotebookMetadata>}
   */
  static getNotebooksMetadataMap() {
    const notebookPaths = this.getNotebooksPaths();
    return notebookPaths.reduce((acc, notebookPath) => {
      const metadata = new NotebookMetadataService(notebookPath)._generateMetadata();
      acc[notebookPath] = metadata;
      return acc;
    }, {});
  }
}
