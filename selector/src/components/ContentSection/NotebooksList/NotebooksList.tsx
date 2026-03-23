import './NotebooksList.scss';

import { IArchivedNotebookMetadata, INotebookMetadata } from '@/shared/notebook-metadata';

import { ArchivedNotebookCard } from './ArchivedNotebookCard/ArchivedNotebookCard';
import { NotebookCard } from './NotebookCard/NotebookCard';

const EmptyNotebooksList = (): JSX.Element => (
  <div className="empty-notebooks-list">
    <span className="spark-font-200">No results found</span>
    <span className="spark-fonr-100">Try adjusting your search or filters</span>
  </div>
);

type NotebooksListProps = {
  items?: INotebookMetadata[];
  archivedItems?: IArchivedNotebookMetadata[];
};

export const NotebooksList = ({ items, archivedItems }: NotebooksListProps): JSX.Element => {
  if (archivedItems) {
    return (
      <div className="notebooks-container">
        {archivedItems.length ? (
          archivedItems.map((notebook) => (
            <ArchivedNotebookCard key={notebook.path} item={notebook}></ArchivedNotebookCard>
          ))
        ) : (
          <EmptyNotebooksList />
        )}
      </div>
    );
  }

  return (
    <div className="notebooks-container">
      {items?.length ? (
        items.map((notebook) => <NotebookCard key={notebook.path} item={notebook}></NotebookCard>)
      ) : (
        <EmptyNotebooksList />
      )}
    </div>
  );
};
