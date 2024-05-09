// @ts-check
/* eslint-env node */

import { execSync } from 'child_process';
import decompress from 'decompress';
import { parse } from 'path';

import { NOTEBOOKS_STATUS_FILE_NAME } from './constants.js';

/**
 * @typedef {{ artifacts: { archive_download_url: string }[] }} ArtifactsResponse
 */

/**
 * @returns {string}
 */
function getLatestNotebooksStatusArtifactUrl() {
  const artifactsResponse = execSync(
    `curl -L "https://api.github.com/repos/openvinotoolkit/openvino_notebooks/actions/artifacts?per_page=1&name=${NOTEBOOKS_STATUS_FILE_NAME}"`
  ).toString();
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const artifactsResponseJson = /** @type {ArtifactsResponse} */ (JSON.parse(artifactsResponse));
  if (!artifactsResponseJson || !artifactsResponseJson?.artifacts?.length) {
    throw Error(
      `Unable to fetch latest artifact "${NOTEBOOKS_STATUS_FILE_NAME}" via GitHub API. Response: ${artifactsResponse}.`
    );
  }
  return artifactsResponseJson.artifacts[0].archive_download_url;
}

/**
 * @param {string} distPath
 * @returns {Promise<void>}
 */
export async function fetchNotebooksStatusFile(distPath) {
  const { GITHUB_TOKEN } = process.env;
  if (!GITHUB_TOKEN) {
    throw Error(`"GITHUB_TOKEN" env varible is not set. Please provide it to fetch notebooks statuses.`);
  }
  console.info(`Fetching latest notebooks status file...`);

  let artifactUrl;
  try {
    artifactUrl = getLatestNotebooksStatusArtifactUrl();
  } catch (error) {
    console.warn(error);
    console.warn('Notebooks status file is not downloaded.');
    return;
  }
  const artifactArchiveFileName = `${parse(NOTEBOOKS_STATUS_FILE_NAME).name}.zip`;
  execSync(
    `curl -H "Accept: application/vnd.github+json" -H "Authorization: token ${GITHUB_TOKEN}" -L --fail -o ${artifactArchiveFileName} "${artifactUrl}"`
  );
  console.info(`Fetched "${artifactArchiveFileName}". Extracting...`);
  await decompress(artifactArchiveFileName, distPath);
  execSync(`rm ${artifactArchiveFileName}`);
  console.info(`Extracted "${artifactArchiveFileName}" to "${distPath}".`);
}
