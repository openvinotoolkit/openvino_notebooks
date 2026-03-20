import './ContentSection.scss';

import { useContext, useEffect, useState } from 'react';

import { Pagination } from '@/components/shared/Pagination/Pagination';
import { isEmbedded } from '@/shared/iframe-detector';
import { sendScrollMessage } from '@/shared/iframe-message-emitter';
import { IArchivedNotebookMetadata, INotebookMetadata } from '@/shared/notebook-metadata';
import { notebooksService, SORT_OPTIONS } from '@/shared/notebooks.service';
import { NotebooksContext } from '@/shared/notebooks-context';

import { ContentSectionHeader } from './ContentSectionHeader/ContentSectionHeader';
import { NotebooksList } from './NotebooksList/NotebooksList';

const notebooksPerPageOptions = [5, 10, 25, 50];

const scrollToTop = () => {
  if (isEmbedded) {
    sendScrollMessage();
  } else {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
};

export const ContentSection = (): JSX.Element => {
  const { selectedTags, searchValue, sort, page, setPage, viewMode, setViewMode } = useContext(NotebooksContext);

  const [notebooks, setNotebooks] = useState<INotebookMetadata[]>([]);
  const [filteredNotebooksCount, setFilteredNotebooksCount] = useState<number>(0);
  const [totalNotebooksCount, setTotalNotebooksCount] = useState<number>(0);

  const [archivedNotebooks, setArchivedNotebooks] = useState<IArchivedNotebookMetadata[]>([]);
  const [filteredArchivedCount, setFilteredArchivedCount] = useState<number>(0);
  const [totalArchivedCount, setTotalArchivedCount] = useState<number>(0);

  const [itemsPerPage, setItemsPerPage] = useState<number>(notebooksPerPageOptions[0]);

  const currentFilteredCount = viewMode === 'active' ? filteredNotebooksCount : filteredArchivedCount;
  const totalPages = Math.ceil(currentFilteredCount / itemsPerPage);

  useEffect(() => {
    setPage(1);
  }, [selectedTags, searchValue, sort, setPage, viewMode]);

  useEffect(() => {
    if (viewMode === 'active') {
      void notebooksService
        .getNotebooks({
          tags: selectedTags,
          searchValue,
          sort,
          offset: (page - 1) * itemsPerPage,
          limit: itemsPerPage,
        })
        .then(([paginatedNotebooks, totalSearchedNotebooks, totalNotebooks]) => {
          setNotebooks(paginatedNotebooks);
          setFilteredNotebooksCount(totalSearchedNotebooks);
          setTotalNotebooksCount(totalNotebooks);
          scrollToTop();
        });
    }
  }, [selectedTags, searchValue, sort, page, itemsPerPage, viewMode]);

  useEffect(() => {
    if (viewMode === 'archived') {
      void notebooksService
        .getArchivedNotebooks({
          searchValue,
          sort,
          offset: (page - 1) * itemsPerPage,
          limit: itemsPerPage,
        })
        .then(([paginatedArchived, totalSearchedArchived, totalArchived]) => {
          setArchivedNotebooks(paginatedArchived);
          setFilteredArchivedCount(totalSearchedArchived);
          setTotalArchivedCount(totalArchived);
          scrollToTop();
        });
    }
  }, [searchValue, sort, page, itemsPerPage, viewMode]);

  // Fetch total archived count once (for tab badge)
  useEffect(() => {
    void notebooksService
      .getArchivedNotebooks({ searchValue: '', sort: SORT_OPTIONS.RECENTLY_ADDED, offset: 0, limit: 0 })
      .then(([, , total]) => setTotalArchivedCount(total));
  }, []);

  const hasItems = viewMode === 'active' ? notebooks.length > 0 : archivedNotebooks.length > 0;

  return (
    <section className="flex-col flex-1 content-section">
      <ContentSectionHeader
        totalActiveCount={totalNotebooksCount}
        filteredCount={currentFilteredCount}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        totalArchivedCount={totalArchivedCount}
      ></ContentSectionHeader>
      {viewMode === 'active' ? (
        <NotebooksList items={notebooks}></NotebooksList>
      ) : (
        <NotebooksList archivedItems={archivedNotebooks}></NotebooksList>
      )}
      {hasItems && (
        <Pagination
          itemsPerPageOptions={notebooksPerPageOptions}
          itemsPerPage={itemsPerPage}
          page={page}
          totalPages={totalPages}
          onChangePage={setPage}
          onChangeItemsPerPage={setItemsPerPage}
        ></Pagination>
      )}
    </section>
  );
};
