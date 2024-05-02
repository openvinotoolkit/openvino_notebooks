// @ts-check
/* eslint-env node */

import { execSync } from 'child_process';
import decompress from 'decompress';
import { parse } from 'path';

/**
 * @typedef {{ artifacts: { archive_download_url: string }[] }} ArtifactsResponse
 */

export const NOTEBOOKS_STATUS_FILE_NAME = 'notebooks-status-map.json';

/**
 * @returns {string}
 */
function getLatestNotebooksStatusesArtifactUrl() {
  const artifactsResponse = execSync(
    // TODO Uncomment after testing
    // `curl -L https://api.github.com/repos/openvinotoolkit/openvino_notebooks/actions/artifacts?per_page=1&name=${NOTEBOOKS_STATUS_FILE_NAME}`
    `curl -L https://api.github.com/repos/yatarkan/openvino_notebooks/actions/artifacts?per_page=1&name=${NOTEBOOKS_STATUS_FILE_NAME}`
  ).toString();
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const artifactsResponseJson = /** @type {ArtifactsResponse} */ (JSON.parse(artifactsResponse));
  if (!artifactsResponseJson || !artifactsResponseJson?.artifacts?.length) {
    throw Error(`Unable to fetch latest artifacts via GitHub API. Response: ${artifactsResponse}.`);
  }
  return artifactsResponseJson.artifacts[0].archive_download_url;
}

/**
 * @param {string} distPath
 * @returns {Promise<void>}
 */
export async function fetchNotebooksStatusesFile(distPath) {
  const { GITHUB_TOKEN } = process.env;
  if (!GITHUB_TOKEN) {
    throw Error(`"GITHUB_TOKEN" env varible is not set. Please provide it to fetch notebooks statuses.`);
  }
  console.info(`Fetching latest notebooks status file...`);

  const artifactUrl = getLatestNotebooksStatusesArtifactUrl();
  const artifactArchiveFileName = `${parse(NOTEBOOKS_STATUS_FILE_NAME).name}.zip`;
  execSync(
    `curl -H "Accept: application/vnd.github+json" -H "Authorization: token ${GITHUB_TOKEN}" -L --fail -o ${artifactArchiveFileName} "${artifactUrl}"`
  );
  console.info(`Fetched "${artifactArchiveFileName}". Extracting...`);
  await decompress(artifactArchiveFileName, distPath);
  execSync(`rm ${artifactArchiveFileName}`);
  console.info(`Extracted "${artifactArchiveFileName}" to "${distPath}".`);
}
