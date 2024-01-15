import './ContentSection.scss';

import { useContext, useEffect, useState } from 'react';

import { INotebookMetadata } from '@/models/notebook-metadata';
import { notebooksService } from '@/models/notebooks.service';
import { NotebooksContext } from '@/models/notebooks-context';

import { ContentSectionHeader } from './ContentSectionHeader/ContentSectionHeader';
import { NotebooksList } from './NotebooksList/NotebooksList';

export const ContentSection = (): JSX.Element => {
  const { selectedTags, searchValue, sort } = useContext(NotebooksContext);

  const [notebooks, setNotebooks] = useState<INotebookMetadata[]>([]);

  const { notebooksTotalCount } = notebooksService;

  useEffect(() => {
    const filteredNotebooks = notebooksService.filterNotebooks({ tags: selectedTags, searchValue, sort });
    setNotebooks(filteredNotebooks);
  }, [selectedTags, searchValue, sort]);

  return (
    <section className="flex-col flex-1 content-section">
      <ContentSectionHeader totalCount={notebooksTotalCount} filteredCount={notebooks.length}></ContentSectionHeader>
      <NotebooksList items={notebooks}></NotebooksList>
    </section>
  );
};
