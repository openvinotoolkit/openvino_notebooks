// @ts-check

import { execSync } from 'child_process';

import { docsNotebooks } from './docs-notebooks.js';
import { NotebookContentReader } from './notebook-content-reader.js';

/** @typedef {import('./notebook-content-reader.js').INotebookMetadata} INotebookMetadata */

export class NotebookMetadataCollector extends NotebookContentReader {
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
    const markdownLinkRegExp = /\[(.+)\]\(.+\)/g;
    return match.groups.title.replace(markdownLinkRegExp, (value, group) => `${group || value}`).trim();
  }

  /**
   * @private
   * @returns {string | null}
   */
  _getImageUrl() {
    const imageUrl = this._getMetadataFromNotebookFile('imageUrl');
    return imageUrl || null;
  }

  /**
   * @private
   * @returns {string}
   */
  _getNotebookCreatedDate() {
    return execSync(
      `git log --follow -1 --pretty=format:"%ad" --date=iso --diff-filter=AC -- ${this._absoluteNotebookPath}`
    ).toString();
  }

  /**
   * @private
   * @returns {string}
   */
  _getNotebookModifiedDate() {
    return execSync(
      `git log -1 --pretty=format:"%ad" --date=iso --diff-filter=a -- ${this._absoluteNotebookPath}`
    ).toString();
  }

  /**
   * @private
   * @returns {string}
   */
  _getNotebookGitHubLink() {
    return `https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/${this._notebookFilePath}`;
  }

  /**
   * @private
   * @returns {string | null}
   */
  _getDocsLink() {
    const { latestDocsNotebooks, latestOVReleaseTag } = docsNotebooks;
    const notebookFileName = this._notebookFileName.replace('.ipynb', '');
    const docsVersion = latestOVReleaseTag.split('.').slice(0, 2)[0];
    const docsUrl = `https://docs.openvino.ai/${docsVersion}/notebooks/${notebookFileName}-with-output.html`;
    return latestDocsNotebooks.includes(this._notebookFilePath) ? docsUrl : null;
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
   * @returns {INotebookMetadata['tags']['libraries']}
   */
  _getLibrariesTags() {
    const codeCells = this._getCodeCells().map(({ source }) => source.join('\n'));
    const tags = [];
    for (const cellContent of codeCells) {
      for (const [tag, patterns] of Object.entries(librariesPatterns)) {
        if (_hasLibraryPattern(cellContent, patterns)) {
          tags.push(tag);
        }
      }
    }
    return [...new Set(tags)];
  }

  /**
   * @private
   * @returns {INotebookMetadata['tags']}
   */
  _getTags() {
    const tags = this._getMetadataFromNotebookFile('tags');
    const libraries = this._getLibrariesTags();
    return {
      categories: [],
      tasks: [],
      other: [],
      ...tags,
      libraries,
    };
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
      path: this._notebookFilePath,
      imageUrl: this._getImageUrl(),
      createdDate: this._getNotebookCreatedDate(),
      modifiedDate: this._getNotebookModifiedDate() || this._getNotebookCreatedDate(),
      links: {
        github: this._getNotebookGitHubLink(),
        docs: this._getDocsLink(),
        colab: this._getNotebookColabLink(),
        binder: this._getNotebookBinderLink(),
      },
      tags: this._getTags(),
    };
  }
}

/** @typedef {typeof import('../shared/notebook-tags.js').LIBRARIES_VALUES} LIBRARIES_VALUES */
/** @typedef {string | { pip: string }} LibraryPattern */
/**
 * A map of library tags to their corresponding patterns used to identify the presence of the library in notebook code cells.
 * Patterns can be strings representing code snippets or objects with a `pip` property for pip install commands.
 *
 * @type {Record<LIBRARIES_VALUES[number], LibraryPattern[]>}
 */
const librariesPatterns = {
  NNCF: ['import nncf', 'from nncf', { pip: 'nncf' }],
  'Model Converter': ['ov.convert_model(', 'openvino.convert_model(', '! ovc'],
  'Model Server': ['import ovmsclient', 'from ovmsclient'],
  'Open Model Zoo': ['omz_downloader', 'omz_converter', 'omz_info_dumper'],
  'Benchmark Tool': ['benchmark_app'],
  'Optimum Intel': [
    'import optimum.intel',
    'from optimum.intel',
    'optimum-cli export openvino',
    'optimum_cli',
    { pip: 'optimum-intel' },
  ],
  Transformers: ['import transformers', 'from transformers', { pip: 'transformers' }],
  Diffusers: ['import diffusers', 'from diffusers', { pip: 'diffusers' }],
  TensorFlow: ['import tensorflow', 'from tensorflow', { pip: 'tensorflow' }],
  'TF Lite': ['.tflite'],
  PyTorch: ['import torch', 'from torch', { pip: 'torch' }],
  ONNX: ['.onnx'],
  PaddlePaddle: ['import paddle', 'from paddle', { pip: 'paddlepaddle' }],
  Ultralytics: ['import ultralytics', 'from ultralytics', { pip: 'ultralytics' }],
  Gradio: ['import gradio', 'from gradio', { pip: 'gradio' }],
  'OpenVINO Tokenizers': ['import openvino_tokenizers', 'from openvino_tokenizers', { pip: 'openvino-tokenizers' }],
  'OpenVINO GenAI': ['import openvino_genai', 'from openvino_genai', { pip: 'openvino-genai' }],
  'OpenVINO Explainable AI': ['import openvino_xai', 'from openvino_xai', { pip: 'openvino-xai' }],
  JAX: ['import jax', 'from jax', { pip: 'jax' }],
  ModelScope: ['import modelscope', 'from modelscope', 'modelscope download'],
};

/**
 * @private
 * @param {string} content
 * @param {LibraryPattern[]} patterns
 * @returns {boolean}
 */
function _hasLibraryPattern(content, patterns) {
  for (const pattern of patterns) {
    if (typeof pattern === 'string' && content.includes(pattern)) {
      return true;
    }
    if (typeof pattern === 'object') {
      const pipInstallRegexp = new RegExp(`^\\s*?%pip\\s+install.*?${pattern.pip}`, 'm');
      const pipInstallHelperRegexp = new RegExp(`^\\s*?pip_install\\([^]*?${pattern.pip}[^]*?\\)`, 'm');
      if (pipInstallRegexp.test(content) || pipInstallHelperRegexp.test(content)) {
        return true;
      }
    }
  }
  return false;
}
