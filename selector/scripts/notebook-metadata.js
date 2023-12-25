// @ts-check

import { argv, exit } from 'process';
import { hideBin } from 'yargs/helpers';
import yargs from 'yargs/yargs';

import { NotebookMetadataService, toMarkdown } from './notebook-metadata-service.js';

yargs(hideBin(argv))
  .command(
    'generate',
    'Generate metadata file',
    {
      file: {
        alias: 'f',
        type: 'string',
        requiresArg: true,
        conflicts: 'all',
      },
      all: {
        alias: 'a',
        type: 'boolean',
        conflicts: 'file',
      },
    },
    ({ all, file }) => {
      if (!all && !file) {
        console.error('Missing required argument: "file" or "all"');
        exit(1);
      }
      const notebooksPaths = all ? NotebookMetadataService.getNotebooksPaths() : file ? [file] : [];
      for (const notebookPath of notebooksPaths) {
        new NotebookMetadataService(notebookPath).generateMetadataFile();
      }
    }
  )
  .command(
    'update',
    'Update metadata file',
    {
      file: {
        alias: 'f',
        type: 'string',
        demandOption: true,
        requiresArg: true,
      },
    },
    ({ file }) => {
      new NotebookMetadataService(file).updateMetadataFile();
    }
  )
  .command(
    'validate',
    'Validate metadata file',
    {
      // TODO Change option to files and accept comma separated files
      file: {
        alias: 'f',
        type: 'string',
        demandOption: true,
        requiresArg: true,
      },
    },
    ({ file }) => {
      try {
        new NotebookMetadataService(file).validateMetadataFile();
      } catch (error) {
        console.error(error.message);
        exit(1);
      }
    }
  )
  .command(
    'to-markdown',
    'Show notebook metadata as markdown',
    {
      file: {
        alias: 'f',
        type: 'string',
        demandOption: true,
        requiresArg: true,
      },
    },
    ({ file }) => {
      const metadata = new NotebookMetadataService(file)._generateMetadata();
      console.info(toMarkdown(metadata));
    }
  )
  .help()
  .parseSync();
