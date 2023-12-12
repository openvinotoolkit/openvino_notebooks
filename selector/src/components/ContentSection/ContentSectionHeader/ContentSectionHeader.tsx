import './ContentSectionHeader.scss';

import { Button } from '@/components/shared/Button/Button';
import { Search } from '@/components/shared/Search/Search';

const sparkClassNames = {
  fontTitleXs: 'spark-font-200',
};

export const ContentSectionHeader = (): JSX.Element => {
  return (
    <div className="content-section-header">
      <div className="flex">
        <h1 className={`${sparkClassNames.fontTitleXs} title`}>Notebooks</h1>
        <span className={`${sparkClassNames.fontTitleXs} counter`}>135</span>
      </div>
      <Button text="Reset Filters" variant="secondary" size="s" className="reset-filters-button"></Button>
      <Search placeholder="Filter notebooks by name" className="notebooks-search"></Search>
    </div>
  );
};
