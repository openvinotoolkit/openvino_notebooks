import './ContentSection.scss';

import { ContentSectionHeader } from './ContentSectionHeader/ContentSectionHeader';
import { NotebooksList } from './NotebooksList/NotebooksList';

export const ContentSection = (): JSX.Element => {
  return (
    <section className="flex-col flex-1 content-section">
      <ContentSectionHeader></ContentSectionHeader>
      <NotebooksList></NotebooksList>
    </section>
  );
};
