import { createContext, Dispatch, SetStateAction, useState } from 'react';

import { INotebookMetadata } from './notebook-metadata';

interface INotebooksSelector {
  selectedTags: INotebookMetadata['tags'];
  setSelectedTags: Dispatch<SetStateAction<INotebooksSelector['selectedTags']>>;
  searchValue: string;
  setSearchValue: Dispatch<SetStateAction<string>>;
  resetFilters: () => void;
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
});

export function useNotebooksSelector(): INotebooksSelector {
  const [selectedTags, setSelectedTags] = useState(defaultSelectedTags);
  const [searchValue, setSearchValue] = useState('');
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
  };
}
