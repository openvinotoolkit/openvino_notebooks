export interface INotebookMetadata {
  title: string;
  description: string;
  additionalResources: object | null;
  imageUrl: string;
  links: {
    github: string;
    colab?: string;
    binder?: string;
  };
  tags: {
    categories: string[];
    tasks: string[];
    models: string[];
    libraries: string[];
    other: string[];
  };
}
