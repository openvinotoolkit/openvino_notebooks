import { createContext, Dispatch, SetStateAction, useState } from 'react';

import { INotebookMetadata } from './notebook-metadata';

interface INotebooksSelector {
  selectedTags: INotebookMetadata['tags'];
  setSelectedTags: Dispatch<SetStateAction<INotebooksSelector['selectedTags']>>;
  searchValue: string;
  setSearchValue: Dispatch<SetStateAction<string>>;
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
});

export function useNotebooksSelector(): INotebooksSelector {
  const [selectedTags, setSelectedTags] = useState(defaultSelectedTags);
  const [searchValue, setSearchValue] = useState('');
  return {
    selectedTags,
    setSelectedTags,
    searchValue,
    setSearchValue,
  };
}
