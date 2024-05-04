// @ts-check

import crypto from 'node:crypto';
import fs from 'node:fs';
import { join, resolve } from 'node:path';

const algorithm = 'sha256';

export const checksumFileName = `checksum.${algorithm}`;

const createSHAHash = () => crypto.createHash(algorithm);

/**
 * @param {string} distPath
 * @returns {Promise<void>}
 */
export async function createBuildChecksumFile(distPath) {
  const paths = await getFilesInDirectory(distPath);
  const fileChecksums = await Promise.all(paths.map(async (path) => await getFileChecksum(path)));
  const buildChecksum = createSHAHash().update(fileChecksums.join('')).digest('hex');
  await fs.promises.writeFile(join(distPath, checksumFileName), buildChecksum, { flag: 'w' });
}

/**
 * @param {string} filePath
 * @returns {Promise<string>}
 */
async function getFileChecksum(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createSHAHash();
    const stream = fs.createReadStream(filePath);
    stream.on('error', (err) => reject(err));
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

/**
 * @param {string} directoryPath
 * @returns {Promise<string[]>}
 */
async function getFilesInDirectory(directoryPath) {
  const entries = await fs.promises.readdir(directoryPath, { withFileTypes: true });
  const entriesPromises = entries.map(async (entry) => {
    const entryPath = resolve(directoryPath, entry.name);
    if (entry.isDirectory()) {
      return await getFilesInDirectory(entryPath);
    }
    return entryPath;
  });
  return (await Promise.all(entriesPromises)).flat(20);
}
