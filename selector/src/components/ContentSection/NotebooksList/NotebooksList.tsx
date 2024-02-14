import './NotebooksList.scss';

import { INotebookMetadata } from '@/shared/notebook-metadata';

import { NotebookCard } from './NotebookCard/NotebookCard';

const EmptyNotebooksList = (): JSX.Element => (
  <div className="empty-notebooks-list">
    <span className="spark-font-200">No results found</span>
    <span className="spark-fonr-100">Try adjusting your search or filters</span>
  </div>
);

type NotebooksListProps = {
  items: INotebookMetadata[];
};

export const NotebooksList = ({ items }: NotebooksListProps): JSX.Element => {
  return (
    <div className="notebooks-container">
      {items.length ? (
        items.map((notebook) => <NotebookCard key={notebook.path} item={notebook}></NotebookCard>)
      ) : (
        <EmptyNotebooksList />
      )}
    </div>
  );
};
