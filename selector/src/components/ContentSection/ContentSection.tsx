import './ContentSection.scss';

import { ContentSectionHeader } from './ContentSectionHeader/ContentSectionHeader';

export const ContentSection = (): JSX.Element => {
  return (
    <section className="flex-col flex-1 content-section">
      <ContentSectionHeader></ContentSectionHeader>
      Content Section
    </section>
  );
};
