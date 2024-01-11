import './NotebooksList.scss';

import { INotebookMetadata } from '@/models/notebook-metadata';

import { NotebookCard } from './NotebookCard/NotebookCard';

type NotebooksListProps = {
  items: INotebookMetadata[];
};

export const NotebooksList = ({ items }: NotebooksListProps): JSX.Element => {
  return (
    <div className="notebooks-container">
      {items.map((notebook, i) => (
        <NotebookCard key={`notebook-${i}`} item={notebook}></NotebookCard>
      ))}
    </div>
  );
};
