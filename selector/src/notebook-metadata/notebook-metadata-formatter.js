// @ts-check

/** @typedef {import('../shared/notebook-metadata.ts').INotebookMetadata} INotebookMetadata */

/**
 *
 * @param {INotebookMetadata} metadata
 * @param {boolean} hasError
 * @returns {string}
 */
export function toMarkdown(metadata, hasError) {
  const { title, imageUrl, path, createdDate, modifiedDate, links, tags } = metadata;
  const markdownLinks = Object.entries(links)
    .filter(([, link]) => link)
    .map(([key, link]) => `[${key}](${link})`);

  /** @type {(tags?: string[]) => string} */
  const toTagsString = (tags) => tags?.map((v) => `\`${v}\``).join(', ') || 'N/A';

  return `
  | Notebook | \`${path}\` |
  | - | - |
  | Valid | ${hasError ? '❌' : '✅'} |
  | Title | ${title} |
  | Image | ${imageUrl ? `<img src="${imageUrl}" height="100">` : 'N/A'} |
  | Created Date | ${createdDate} |
  | Modified Date | ${modifiedDate} |
  | Links | ${markdownLinks.join(', ')} |
  | **Tags:** | |
  | Categories | ${toTagsString(tags?.categories)} |
  | Tasks | ${toTagsString(tags?.tasks)} |
  | Libraries | ${toTagsString(tags?.libraries)} |
  | Common | ${toTagsString(tags?.other)} |
`;
}
