import { INotebookMetadata } from './notebook-metadata';

interface INotebooksFilters {
  tags: INotebookMetadata['tags'];
  searchValue: string;
}

class NotebooksService {
  static async loadNotebooks(): Promise<NotebooksService> {
    const notebooksMap = (await fetch(`/notebooks-metadata-map.json`).then((response) => response.json())) as Record<
      string,
      INotebookMetadata
    >;
    return new NotebooksService(notebooksMap);
  }

  constructor(private _notebooksMap: Record<string, INotebookMetadata>) {}

  get notebooks(): INotebookMetadata[] {
    return Object.values(this._notebooksMap);
  }

  get notebooksTotalCount(): number {
    return Object.keys(this._notebooksMap).length;
  }

  filterNotebooks({ tags, searchValue }: INotebooksFilters): INotebookMetadata[] {
    return this.notebooks
      .filter((notebook) => {
        const flatNotebookTags = Object.values(notebook.tags).flat();
        const flatSelectedTags = Object.values(tags).flat();

        return flatSelectedTags.every((tag) => flatNotebookTags.includes(tag));
      })
      .filter(({ title }) => title.toLowerCase().includes(searchValue.toLowerCase()));
  }
}

export const notebooksService = await NotebooksService.loadNotebooks();
