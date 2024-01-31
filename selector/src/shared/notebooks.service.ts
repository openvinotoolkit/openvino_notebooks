import { INotebookMetadata } from './notebook-metadata';

export const SORT_OPTIONS = {
  DEFAULT: 'Default',
  RECENTLY_ADDED: 'Recently Added',
  RECENTLY_UPDATED: 'Recently Updated',
  NAME_ASCENDING: 'Name (Ascending)',
  NAME_DESCENDING: 'Name (Descending)',
} as const;

export type SortValues = (typeof SORT_OPTIONS)[keyof typeof SORT_OPTIONS];

interface INotebooksFilters {
  tags: INotebookMetadata['tags'];
  searchValue: string;
  sort: SortValues;
  offset: number;
  limit: number;
}

class NotebooksService {
  static async loadNotebooks(): Promise<NotebooksService> {
    const { BASE_URL } = import.meta.env;
    const notebooksMap = (await fetch(`${BASE_URL}notebooks-metadata-map.json`).then((response) =>
      response.json()
    )) as Record<string, INotebookMetadata>;
    return new NotebooksService(notebooksMap);
  }

  constructor(private _notebooksMap: Record<string, INotebookMetadata>) {}

  private get _notebooks(): INotebookMetadata[] {
    return Object.values(this._notebooksMap);
  }

  get notebooksTotalCount(): number {
    return Object.keys(this._notebooksMap).length;
  }

  getNotebooks({ tags, searchValue, sort, offset, limit }: INotebooksFilters): [INotebookMetadata[], number] {
    const filteredNotebooks = this._notebooks
      .filter((notebook) => {
        const flatNotebookTags = Object.values(notebook.tags).flat();
        const flatSelectedTags = Object.values(tags).flat();

        return flatSelectedTags.every((tag) => flatNotebookTags.includes(tag));
      })
      .filter(({ title }) => title.toLowerCase().includes(searchValue.toLowerCase()));
    const sortedPaginatedNotebooks = filteredNotebooks.sort(this._getCompareFn(sort)).slice(offset, offset + limit);
    return [sortedPaginatedNotebooks, filteredNotebooks.length];
  }

  private _getCompareFn(sort: SortValues): Parameters<Array<INotebookMetadata>['sort']>[0] {
    if (sort === SORT_OPTIONS.RECENTLY_ADDED) {
      return (a: INotebookMetadata, b: INotebookMetadata) =>
        new Date(b.createdDate).getTime() - new Date(a.createdDate).getTime();
    }
    if (sort === SORT_OPTIONS.RECENTLY_UPDATED) {
      return (a: INotebookMetadata, b: INotebookMetadata) =>
        new Date(b.modifiedDate).getTime() - new Date(a.modifiedDate).getTime();
    }
    if (sort === SORT_OPTIONS.NAME_ASCENDING) {
      return (a: INotebookMetadata, b: INotebookMetadata) => a.title.toUpperCase().localeCompare(b.title.toUpperCase());
    }
    if (sort === SORT_OPTIONS.NAME_DESCENDING) {
      return (a: INotebookMetadata, b: INotebookMetadata) => b.title.toUpperCase().localeCompare(a.title.toUpperCase());
    }
    if (sort === SORT_OPTIONS.DEFAULT) {
      return (a: INotebookMetadata, b: INotebookMetadata) => b.path.toUpperCase().localeCompare(a.path.toUpperCase());
    }
  }
}

export const notebooksService = await NotebooksService.loadNotebooks();
