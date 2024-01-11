import { INotebookMetadata } from './notebook-metadata';

class NotebooksService {
  static async loadNotebooks(): Promise<NotebooksService> {
    const notebooksMap = (await fetch('/notebooks-metadata-map.json').then((response) => response.json())) as Record<
      string,
      INotebookMetadata
    >;
    return new NotebooksService(notebooksMap);
  }

  constructor(private _notebooksMap: Record<string, INotebookMetadata>) {}

  get notebooks(): INotebookMetadata[] {
    return Object.values(this._notebooksMap);
  }
}

export const notebooksService = await NotebooksService.loadNotebooks();
