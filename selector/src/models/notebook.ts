import { CATEGORIES } from './notebook-tags.js';

type ObjectValues<T> = T[keyof T];

// TODO Consider adding notebook relative path
export interface INotebookMetadata {
  title: string;
  description: string;
  imageUrl: string | null;
  createdDate: string;
  modifiedDate: string;
  links: {
    github: string;
    colab: string | null;
    binder: string | null;
  };
  tags: {
    categories: ObjectValues<typeof CATEGORIES>[];
    tasks: string[];
    libraries: string[];
    other: string[];
  };
}
