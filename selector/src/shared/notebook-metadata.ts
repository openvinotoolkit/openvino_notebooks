import { CATEGORIES, TASKS_VALUES } from './notebook-tags.js';

type ObjectValues<T> = T[keyof T];

export interface INotebookMetadata {
  title: string;
  path: string;
  imageUrl: string | null;
  createdDate: string;
  modifiedDate: string;
  links: {
    github: string;
    docs: string | null;
    colab: string | null;
    binder: string | null;
  };
  tags: {
    categories: ObjectValues<typeof CATEGORIES>[];
    tasks: typeof TASKS_VALUES;
    libraries: string[];
    other: string[];
  };
}

export interface IArchivedNotebookMetadata {
  title: string;
  path: string;
  imageUrl: string | null;
  lastBranch: string;
  githubUrl: string;
  tags: {
    categories: string[];
    tasks: string[];
    libraries: string[];
    other: string[];
  };
}
