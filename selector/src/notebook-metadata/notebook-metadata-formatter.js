// @ts-check

/** @typedef {import('../models/notebook').INotebookMetadata} INotebookMetadata */

/**
 *
 * @param {INotebookMetadata} metadata
 * @returns {string}
 */
export function toMarkdown(metadata) {
  const { title, imageUrl, path, createdDate, modifiedDate, links, tags } = metadata;
  const markdownLinks = Object.entries(links)
    .filter(([, link]) => link)
    .map(([key, link]) => `[${key}](${link})`);

  /** @type {(tags: string[]) => string} */
  const toTagsString = (tags) => tags.map((v) => `\`${v}\``).join(', ');

  return `
  | Notebook | \`${path}\` |
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
  | Common | ${toTagsString(tags.other)} |
  `;
}
