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
   * Extracts HuggingFace-style model ids (`org/model`) from a text blob.
   *
   * @private
   * @param {string} text
   * @returns {string[]}
   */
  _extractHfIds(text) {
    const ids = [];
    const re = /["']([A-Za-z0-9][\w.-]*\/[\w.-]+)["']/g;
    let match;
    while ((match = re.exec(text)) !== null) {
      const id = match[1];
      // Skip file paths (e.g. `data/coco.mp4`, `images/cat.png`).
      if (FILE_EXTENSION_RE.test(id)) {
        continue;
      }
      ids.push(id);
    }
    return ids;
  }

  /**
   * Extracts well-known model names written without an `org/` prefix (e.g. Ultralytics
   * YOLO names like `yolo11n` or `yolov8n-seg`), which the HuggingFace-id extractor misses.
   *
   * @private
   * @param {string} text
   * @returns {string[]}
   */
  _extractKnownModelNames(text) {
    const names = [];
    for (const re of KNOWN_MODEL_NAME_PATTERNS) {
      const globalRe = re.flags.includes('g') ? re : new RegExp(re.source, `${re.flags}g`);
      globalRe.lastIndex = 0;
      let match;
      while ((match = globalRe.exec(text)) !== null) {
        names.push(match[0].toLowerCase());
      }
    }
    return names;
  }

  /**
   * Extracts a balanced `{ ... }` dict block assigned to the given symbol.
   *
   * @private
   * @param {string} text
   * @param {string} symbol
   * @returns {string | null}
   */
  _extractDictBlock(text, symbol) {
    const startRe = new RegExp(`${symbol}\\s*=\\s*{`);
    const startMatch = startRe.exec(text);
    if (!startMatch) {
      return null;
    }
    const openIndex = text.indexOf('{', startMatch.index);
    let depth = 0;
    for (let i = openIndex; i < text.length; i++) {
      if (text[i] === '{') {
        depth++;
      } else if (text[i] === '}') {
        depth--;
        if (depth === 0) {
          return text.slice(openIndex, i + 1);
        }
      }
    }
    return null;
  }

  /**
   * Returns the innermost balanced `{ ... }` block that surrounds the given position.
   *
   * @private
   * @param {string} text
   * @param {number} index
   * @returns {string | null}
   */
  _enclosingBraceBlock(text, index) {
    let depth = 0;
    let openIndex = -1;
    for (let i = index; i >= 0; i--) {
      if (text[i] === '}') {
        depth++;
      } else if (text[i] === '{') {
        if (depth === 0) {
          openIndex = i;
          break;
        }
        depth--;
      }
    }
    if (openIndex === -1) {
      return null;
    }
    depth = 0;
    for (let i = openIndex; i < text.length; i++) {
      if (text[i] === '{') {
        depth++;
      } else if (text[i] === '}') {
        depth--;
        if (depth === 0) {
          return text.slice(openIndex, i + 1);
        }
      }
    }
    return null;
  }

  /**
   * Extracts `model_id` values from a `SUPPORTED_*_MODELS` dict block.
   * When `requiredKey` is set, only entries whose config also defines that key are included
   * (e.g. RAG notebooks keep only models that provide a `rag_prompt_template`).
   *
   * @private
   * @param {string} blockText
   * @param {string | null} [requiredKey]
   * @returns {string[]}
   */
  _extractConfigModelIds(blockText, requiredKey = null) {
    const ids = [];
    const re = /["']model_id["']\s*:\s*["']([^"']+)["']/g;
    let match;
    while ((match = re.exec(blockText)) !== null) {
      if (requiredKey) {
        const configBlock = this._enclosingBraceBlock(blockText, match.index);
        if (!configBlock || !configBlock.includes(`"${requiredKey}"`)) {
          continue;
        }
      }
      ids.push(match[1]);
    }
    return ids;
  }

  /**
   * Whether the selection-widget helper is called relying on its default (shared) model set,
   * i.e. at least one call does not pass its own `models=` dict.
   *
   * @private
   * @param {string} codeText
   * @param {string} widgetFn
   * @returns {boolean}
   */
  _widgetUsesSharedSet(codeText, widgetFn) {
    const callToken = `${widgetFn}(`;
    let searchIndex = codeText.indexOf(callToken);
    while (searchIndex !== -1) {
      const openIndex = searchIndex + callToken.length - 1;
      let depth = 0;
      let closeIndex = -1;
      for (let i = openIndex; i < codeText.length; i++) {
        if (codeText[i] === '(') {
          depth++;
        } else if (codeText[i] === ')') {
          depth--;
          if (depth === 0) {
            closeIndex = i;
            break;
          }
        }
      }
      const callArgs = closeIndex === -1 ? codeText.slice(openIndex) : codeText.slice(openIndex, closeIndex + 1);
      if (!callArgs.includes('models=')) {
        return true;
      }
      searchIndex = codeText.indexOf(callToken, closeIndex === -1 ? searchIndex + callToken.length : closeIndex);
    }
    return false;
  }

  /**
   * Collects model names referenced by the notebook, both inline in code cells
   * and from the symbol-specific `SUPPORTED_*_MODELS` dictionaries.
   *
   * @private
   * @returns {INotebookMetadata['models']}
   */
  _getModels() {
    /** @type {Set<string>} */
    const models = new Set();

    const codeCellsText = this._getCodeCells()
      .map(({ source }) => source.join(''))
      .join('\n');
    const siblingText = this._getNotebookDirPyFilesContent().join('\n');
    const sharedConfigText = this._getSharedLlmConfigContent() || '';

    // Inline model ids written directly in the notebook code cells.
    for (const id of this._extractHfIds(codeCellsText)) {
      models.add(id);
    }

    // Well-known model names written without an `org/` prefix (e.g. YOLO).
    for (const name of this._extractKnownModelNames(codeCellsText)) {
      models.add(name);
    }

    // Config-based ids: only include the model set(s) the notebook actually uses.
    // A local sibling config is preferred over the shared utils/llm_config.py.
    for (const { symbol, widgetFn } of MODEL_CONFIG_SYMBOLS) {
      const isUsed = codeCellsText.includes(symbol) || (!!widgetFn && this._widgetUsesSharedSet(codeCellsText, widgetFn));
      if (!isUsed) {
        continue;
      }
      const block = this._extractDictBlock(siblingText, symbol) || this._extractDictBlock(sharedConfigText, symbol);
      if (!block) {
        continue;
      }
      // RAG notebooks filter LLM models to those defining a `rag_prompt_template`.
      const requiredKey =
        symbol === 'SUPPORTED_LLM_MODELS' && codeCellsText.includes('rag_prompt_template')
          ? 'rag_prompt_template'
          : null;
      for (const id of this._extractConfigModelIds(block, requiredKey)) {
        models.add(id);
      }
    }

    // Normalize: add the short name (without org prefix) alongside the full id.
    /** @type {Set<string>} */
    const result = new Set();
    for (const id of models) {
      result.add(id);
      const shortName = id.split('/').pop();
      if (shortName) {
        result.add(shortName);
      }
    }
    return [...result].sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
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
      models: this._getModels(),
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
 * Matches file-path-like tokens ending with a known media/asset extension, to exclude them from model ids.
 */
const FILE_EXTENSION_RE =
  /\.(mp4|avi|mov|mkv|webm|png|jpe?g|gif|bmp|webp|svg|mp3|wav|flac|ogg|xml|bin|json|txt|pdf|csv|tsv|npy|npz|onnx|safetensors|pt|pth|md|yaml|yml|zip|tar|gz|html?)$/i;

/**
 * Patterns for well-known model names that appear in notebooks without an `org/` prefix and
 * are therefore not caught by the HuggingFace-id extractor. Extend this list for new families.
 *
 * @type {RegExp[]}
 */
const KNOWN_MODEL_NAME_PATTERNS = [
  // Ultralytics YOLO family: yolo11n, yolov8n, yolo26, yolo11x-seg, yolov9c-pose, ...
  /\byolo(?:v)?\d+[a-z]?(?:-(?:seg|pose|cls|obb))?\b/gi,
];

/**
 * Maps each `SUPPORTED_*_MODELS` dictionary to the code signals that indicate a notebook uses it.
 * A notebook uses the shared (disjoint) set only when it references the symbol directly, or calls the
 * selection widget without passing its own `models=` dict. `widgetFn` is `null` when no such helper exists.
 *
 * @type {{ symbol: string, widgetFn: string | null }[]}
 */
const MODEL_CONFIG_SYMBOLS = [
  { symbol: 'SUPPORTED_VLM_MODELS', widgetFn: 'get_vlm_selection_widget' },
  { symbol: 'SUPPORTED_LLM_MODELS', widgetFn: 'get_llm_selection_widget' },
  { symbol: 'SUPPORTED_EMBEDDING_MODELS', widgetFn: null },
  { symbol: 'SUPPORTED_RERANK_MODELS', widgetFn: null },
];

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
