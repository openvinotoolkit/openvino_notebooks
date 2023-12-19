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
    categories: string[];
    tasks: string[];
    models: string[];
    libraries: string[];
    other: string[];
  };
}
