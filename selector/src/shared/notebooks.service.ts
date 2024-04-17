import { INotebookMetadata } from './notebook-metadata';

export const SORT_OPTIONS = {
  RECENTLY_ADDED: 'Recently Added',
  RECENTLY_UPDATED: 'Recently Updated',
  NAME_ASCENDING: 'Name (A-Z)',
  NAME_DESCENDING: 'Name (Z-A)',
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
  private _notebooksMap: Record<string, INotebookMetadata> | null = null;

  private async _getNotebooksMap(): Promise<Record<string, INotebookMetadata>> {
    if (!this._notebooksMap) {
      const { BASE_URL } = import.meta.env;
      const notebooksMap = (await fetch(`${BASE_URL}notebooks-metadata-map.json`).then((response) =>
        response.json()
      )) as Record<string, INotebookMetadata>;
      this._notebooksMap = notebooksMap;
    }
    return this._notebooksMap;
  }

  async getNotebooks({
    tags,
    searchValue,
    sort,
    offset,
    limit,
  }: INotebooksFilters): Promise<[INotebookMetadata[], number, number]> {
    const notebooks = Object.values(await this._getNotebooksMap());
    const filteredNotebooks = notebooks
      .filter((notebook) => {
        const flatNotebookTags = Object.values(notebook.tags).flat();
        const flatSelectedTags = Object.values(tags).flat();

        return flatSelectedTags.every((tag) => flatNotebookTags.includes(tag));
      })
      .filter(({ title }) => title.toLowerCase().includes(searchValue.toLowerCase()));
    const sortedPaginatedNotebooks = filteredNotebooks.sort(this._getCompareFn(sort)).slice(offset, offset + limit);
    return [sortedPaginatedNotebooks, filteredNotebooks.length, notebooks.length];
  }

  async getOtherTags(): Promise<string[]> {
    const notebooks = Object.values(await this._getNotebooksMap());
    return notebooks
      .reduce((acc, { tags }) => {
        for (const tag of tags.other) {
          if (!acc.includes(tag)) {
            acc.push(tag);
          }
        }
        return acc;
      }, [] as string[])
      .sort((a, b) => a.toUpperCase().localeCompare(b.toUpperCase()));
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
  }
}

export const notebooksService = new NotebooksService();
