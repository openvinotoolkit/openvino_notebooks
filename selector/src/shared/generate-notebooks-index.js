// @ts-check

import fs from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const NOTEBOOKS_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'notebooks');
const INDEX_FILE_NAME = 'README.md';

/** @typedef {import('./notebook-metadata.ts').INotebookMetadata} INotebookMetadata */

/**
 * @param {Record<string, INotebookMetadata>} notebooksMetadataMap
 * @returns {Record<string, INotebookMetadata[]>}
 */
function getCategoryToNotebooksMetadataMap(notebooksMetadataMap) {
  /** @type {Record<string, INotebookMetadata[]>} */
  const categoryToNotebooksMetadataMap = {};
  return Object.values(notebooksMetadataMap).reduce((acc, notebookMetadata) => {
    const { categories } = notebookMetadata.tags;
    for (const category of categories) {
      if (!acc[category]) {
        acc[category] = [notebookMetadata];
      } else {
        acc[category].push(notebookMetadata);
      }
    }
    return acc;
  }, categoryToNotebooksMetadataMap);
}

/**
 * @param {Record<string, INotebookMetadata[]>} categoryToNotebooksMetadataMap
 * @returns {string}
 */
function formatToIndexMarkdown(categoryToNotebooksMetadataMap) {
  const mdContents = [
    '# OpenVINO Notebooks\n\n',
    '[OpenVINO™ Notebooks at GitHub Pages](https://openvinotoolkit.github.io/openvino_notebooks/)\n\n',
  ];

  for (const [category, notebooksMetadataList] of Object.entries(categoryToNotebooksMetadataMap).sort()) {
    mdContents.push(`## ${category}\n\n`);
    const notebooksListItems = notebooksMetadataList.map(({ title, path }) => `- [${title}](./${path})\n`);
    mdContents.push(...notebooksListItems, '\n');
  }

  return mdContents.join('');
}

/**
 * @param {string} notebooksMetadataMapFilePath
 * @returns {Promise<void>}
 */
export async function generateNotebooksIndex(notebooksMetadataMapFilePath) {
  const notebooksMetadataMapContent = await fs.promises.readFile(notebooksMetadataMapFilePath, { encoding: 'utf8' });

  /** @type {Record<string, INotebookMetadata>} */
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const notebooksMetadataMap = JSON.parse(notebooksMetadataMapContent);

  const categoryToNotebooksMetadataMap = getCategoryToNotebooksMetadataMap(notebooksMetadataMap);

  const indexContent = formatToIndexMarkdown(categoryToNotebooksMetadataMap);

  await fs.promises.writeFile(join(NOTEBOOKS_DIR, INDEX_FILE_NAME), indexContent, { flag: 'w' });
}
