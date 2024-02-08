import { createContext, Dispatch, SetStateAction, useState } from 'react';

import { INotebookMetadata } from './notebook-metadata';
import { SORT_OPTIONS, SortValues } from './notebooks.service';

interface INotebooksSelector {
  selectedTags: INotebookMetadata['tags'];
  setSelectedTags: Dispatch<SetStateAction<INotebooksSelector['selectedTags']>>;
  searchValue: string;
  setSearchValue: Dispatch<SetStateAction<INotebooksSelector['searchValue']>>;
  resetFilters: () => void;
  sort: SortValues;
  setSort: Dispatch<SetStateAction<INotebooksSelector['sort']>>;
  page: number;
  setPage: Dispatch<SetStateAction<INotebooksSelector['page']>>;
}

const defaultSelectedTags: INotebookMetadata['tags'] = {
  categories: [],
  tasks: [],
  libraries: [],
  other: [],
};

export const NotebooksContext = createContext<INotebooksSelector>({
  selectedTags: defaultSelectedTags,
  setSelectedTags: () => {},
  searchValue: '',
  setSearchValue: () => {},
  resetFilters: () => {},
  sort: SORT_OPTIONS.RECENTLY_ADDED,
  setSort: () => {},
  page: 1,
  setPage: () => {},
});

export function useNotebooksSelector(): INotebooksSelector {
  const [selectedTags, setSelectedTags] = useState(defaultSelectedTags);
  const [searchValue, setSearchValue] = useState('');
  const [sort, setSort] = useState<SortValues>(SORT_OPTIONS.RECENTLY_ADDED);
  const [page, setPage] = useState<number>(1);

  const resetFilters = () => {
    setSelectedTags(defaultSelectedTags);
    setSearchValue('');
  };

  return {
    selectedTags,
    setSelectedTags,
    searchValue,
    setSearchValue,
    resetFilters,
    sort,
    setSort,
    page,
    setPage,
  };
}
