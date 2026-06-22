import { ARCHIVED_NOTEBOOKS_FILE_NAME, NOTEBOOKS_METADATA_FILE_NAME, NOTEBOOKS_STATUS_FILE_NAME } from './constants';
import { IArchivedNotebookMetadata, INotebookMetadata } from './notebook-metadata';
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

interface IArchivedNotebooksFilters {
  searchValue: string;
  sort: SortValues;
  offset: number;
  limit: number;
}

type NotebooksMap = Record<string, INotebookMetadata & { status?: INotebookStatus['status'] }>;

export type NotebookItem = NotebooksMap[string];

class NotebooksService {
  private _notebooksMap: NotebooksMap | null = null;
  private _allNotebooksTags: string[] = [];
  private _archivedNotebooks: IArchivedNotebookMetadata[] | null = null;

  private async _getNotebooksMap(): Promise<NotebooksMap> {
    if (!this._notebooksMap) {
      const { BASE_URL } = import.meta.env;

      const notebooksMetadataMap = (await fetch(`${BASE_URL}${NOTEBOOKS_METADATA_FILE_NAME}`).then((response) =>
        response.json()
      )) as Record<string, INotebookMetadata>;

      const notebooksStatusMap = (await fetch(`${BASE_URL}${NOTEBOOKS_STATUS_FILE_NAME}`).then((response) =>
        response.ok ? response.json() : {}
      )) as Record<string, INotebookStatus>;

      this._notebooksMap = this._getNotebooksMapWithStatuses(notebooksMetadataMap, notebooksStatusMap);
      this._allNotebooksTags = this._getAllNotebooksTags(this._notebooksMap);
    }
    return this._notebooksMap;
  }

  private async _getArchivedNotebooks(): Promise<IArchivedNotebookMetadata[]> {
    if (!this._archivedNotebooks) {
      const { BASE_URL } = import.meta.env;
      try {
        const response = await fetch(`${BASE_URL}${ARCHIVED_NOTEBOOKS_FILE_NAME}`);
        this._archivedNotebooks = response.ok ? ((await response.json()) as IArchivedNotebookMetadata[]) : [];
      } catch {
        this._archivedNotebooks = [];
      }
    }
    return this._archivedNotebooks;
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

  get allNotebooksTags(): typeof this._allNotebooksTags {
    return this._allNotebooksTags;
  }

  async getArchivedNotebooks({
    searchValue,
    sort,
    offset,
    limit,
  }: IArchivedNotebooksFilters): Promise<[IArchivedNotebookMetadata[], number, number]> {
    const archived = await this._getArchivedNotebooks();
    const normalizedSearch = searchValue.trim().toLowerCase();
    const filtered =
      normalizedSearch === '' ? archived : archived.filter(({ title }) => title.toLowerCase().includes(normalizedSearch));
    if (limit === 0) {
      return [[], filtered.length, archived.length];
    }
    const sorted = filtered.sort(this._getArchivedCompareFn(sort)).slice(offset, offset + limit);
    return [sorted, filtered.length, archived.length];
  }

  private _getArchivedCompareFn(
    sort: SortValues
  ): Parameters<Array<IArchivedNotebookMetadata>['sort']>[0] {
    if (sort === SORT_OPTIONS.RECENTLY_ADDED || sort === SORT_OPTIONS.RECENTLY_UPDATED) {
      // Sort by lastBranch descending (newest release first)
      return (a, b) => b.lastBranch.localeCompare(a.lastBranch, undefined, { numeric: true });
    }
    if (sort === SORT_OPTIONS.NAME_ASCENDING) {
      return (a, b) => a.title.toUpperCase().localeCompare(b.title.toUpperCase());
    }
    if (sort === SORT_OPTIONS.NAME_DESCENDING) {
      return (a, b) => b.title.toUpperCase().localeCompare(a.title.toUpperCase());
    }
    return () => 0;
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
    const result: NotebooksMap = { ...metadataMap };
    for (const [notebookPath, { status }] of Object.entries(statusMap)) {
      if (!result[notebookPath]) {
        console.warn(`Unknown notebook "${notebookPath}" found in status report.`);
        continue;
      }
      if (result[notebookPath].status) {
        console.warn(`Status of the notebook "${notebookPath}" already exists and will be overrided.`);
      }
      result[notebookPath].status = status;
    }
    return result;
  }

  private _getAllNotebooksTags(notebooksMap: NotebooksMap): string[] {
    const tagsSet = Object.values(notebooksMap).reduce((acc, { tags }) => {
      const notebookTags = Object.values(tags).flat();
      return new Set([...acc, ...notebookTags]);
    }, new Set<string>());
    return [...tagsSet];
  }
}

export const notebooksService = new NotebooksService();
