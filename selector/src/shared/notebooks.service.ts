import { INotebookMetadata } from './notebook-metadata';
import { INotebookStatus } from './notebook-status';

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

type NotebooksMap = Record<string, INotebookMetadata & { status?: INotebookStatus['status'] }>;

export type NotebookItem = NotebooksMap[string];

class NotebooksService {
  private _notebooksMap: NotebooksMap | null = null;

  private async _getNotebooksMap(): Promise<NotebooksMap> {
    if (!this._notebooksMap) {
      const { BASE_URL } = import.meta.env;
      const notebooksMetadataMap = (await fetch(`${BASE_URL}notebooks-metadata-map.json`).then((response) =>
        response.json()
      )) as Record<string, INotebookMetadata>;
      // TODO Consider reusing filename
      const notebooksStatusMap = (await fetch(`${BASE_URL}notebooks-status-map.json`).then((response) =>
        response.json()
      )) as Record<string, INotebookStatus>;
      this._notebooksMap = this._getNotebooksMapWithStatuses(notebooksMetadataMap, notebooksStatusMap);
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

  private _getNotebooksMapWithStatuses(
    metadataMap: Record<string, INotebookMetadata>,
    statusMap: Record<string, INotebookStatus>
  ): NotebooksMap {
    const result = { ...metadataMap } as NotebooksMap;
    const notebooksKeys = Object.keys(result);
    for (const [key, { status }] of Object.entries(statusMap)) {
      // TODO Unify keys in both maps to prevent searching similar keys
      const notebookKey = notebooksKeys.find((v) => v.includes(key));
      if (!notebookKey || !result[notebookKey]) {
        continue;
      }
      result[notebookKey].status = status;
    }
    return result;
  }
}

export const notebooksService = new NotebooksService();
