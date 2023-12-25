// @ts-check

import { execSync } from 'child_process';
import { existsSync, readFileSync, writeFileSync } from 'fs';
import { globSync } from 'glob';
import { basename, dirname, join } from 'path';
import { fileURLToPath } from 'url';

import { validateNotebookMetadata } from './notebook-metadata-validators.js';

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

/**
 * Returns path to metadata file releated to the notebook path
 *
 * @param {string} notebookPath
 * @returns {string}
 */
function getNotebookMetadataFilePath(notebookPath) {
  const metadataFileName = `${basename(notebookPath, '.ipynb')}.metadata.json`;
  return join(NOTEBOOKS_DIRECTORY_PATH, dirname(notebookPath), metadataFileName);
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
    /** @private */
    this._metadataFilePath = getNotebookMetadataFilePath(this._notebookFilePath);

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
   * Creates metadata file for corresponding notebook
   *
   * @returns {void}
   */
  generateMetadataFile() {
    console.info(`Generating metadata file for notebook "${this._notebookFilePath}"...`);
    const metadata = this._generateMetadata();
    writeFileSync(this._metadataFilePath, JSON.stringify(metadata, null, 2), { flag: 'w' });
    console.info(`Metadata file "${this._metadataFilePath}" is generated.`);
  }

  /**
   * Updates metadata file for corresponding notebook
   *
   * @returns {void}
   */
  updateMetadataFile() {
    console.info(`Updating metadata file "${this._metadataFilePath}"...`);
    /** @type {INotebookMetadata} */
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const metadata = JSON.parse(readFileSync(this._metadataFilePath, { encoding: 'utf8' }));
    metadata.modifiedDate = this._getNotebookModifiedDate();
    writeFileSync(this._metadataFilePath, JSON.stringify(metadata, null, 2), { flag: 'w' });
    console.info(`Metadata file is updated.`);
  }

  /**
   * Validates metadata file for corresponding notebook
   *
   * @returns {void}
   */
  validateMetadataFile() {
    console.info(`Validating metadata file "${this._metadataFilePath}"...`);
    /** @type {INotebookMetadata} */
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const metadata = JSON.parse(readFileSync(this._metadataFilePath, { encoding: 'utf8' }));
    try {
      validateNotebookMetadata(metadata);
    } catch (error) {
      throw Error(`Invalid metadata file "${this._metadataFilePath}".\n${error}`);
    }
    console.info(`Metadata file is valid.`);
  }

  static getNotebooksPaths() {
    return globSync('**/*.ipynb', {
      ignore: ['**/.ipynb_checkpoints/*', '**/notebook_utils.ipynb'],
      cwd: NOTEBOOKS_DIRECTORY_PATH,
    });
  }

  /**
   * Returns notebook metadata object by notebook path
   *
   * @param {string} notebookFilePath
   * @private
   * @returns {INotebookMetadata}
   */
  static _getNotebookMetadata(notebookFilePath) {
    const metadataFilePath = getNotebookMetadataFilePath(notebookFilePath);
    const absoluteMetadataFilePath = join(NOTEBOOKS_DIRECTORY_PATH, metadataFilePath);

    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return JSON.parse(readFileSync(absoluteMetadataFilePath, { encoding: 'utf8' }));
  }

  /**
   * Returns map with metadata objects for all notebooks
   *
   * @returns {Record<string, INotebookMetadata>}
   */
  static getNotebooksMetadataMap() {
    const notebookPaths = this.getNotebooksPaths();
    return notebookPaths.reduce((acc, notebookPath) => {
      acc[notebookPath] = this._getNotebookMetadata(notebookPath);
      return acc;
    }, {});
  }
}

/**
 *
 * @param {INotebookMetadata} metadata
 * @returns {string}
 */
export function toMarkdown(metadata) {
  const { title, imageUrl, createdDate, modifiedDate, links, tags } = metadata;
  const markdownLinks = Object.entries(links)
    .filter(([, link]) => link)
    .map(([key, link]) => `[${key}](${link})`);

  /** @type {(tags: string[]) => string} */
  const toTagsString = (tags) => tags.map((v) => `\`${v}\``).join(', ');

  return `
  | Notebook | \`./001-hello-world/001-hello-world.ipynb\` |
  | - | - |
  | Title | ${title} |
  | Image | <img src="${imageUrl}"  height="100"> |
  | Created Date | ${createdDate} |
  | Modified Date | ${modifiedDate} |
  | Links | ${markdownLinks.join(', ')} |
  | **Tags:** | |
  | Categories | ${toTagsString(tags.categories)} |
  | Tasks | ${toTagsString(tags.tasks)} |
  | Libraries | ${toTagsString(tags.libraries)} |
  | Common | ${toTagsString(tags.other)} |`;
}
