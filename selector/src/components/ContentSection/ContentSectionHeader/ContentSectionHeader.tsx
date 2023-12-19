import './ContentSectionHeader.scss';

import { Button } from '@/components/shared/Button/Button';
import { Search } from '@/components/shared/Search/Search';

const sparkClassNames = {
  fontTitleXs: 'spark-font-200',
};

type ContentSectionHeaderProps = {
  totalCount: number;
  filteredCount: number;
  showResetFilters?: boolean;
  onSearch?: (value: string) => void;
};

export const ContentSectionHeader = ({
  totalCount,
  filteredCount,
  onSearch,
}: ContentSectionHeaderProps): JSX.Element => {
  const isFiltered = filteredCount !== totalCount;

  return (
    <div className="content-section-header">
      <div className="flex">
        <h1 className={`${sparkClassNames.fontTitleXs} title`}>Notebooks</h1>
        <span className={`${sparkClassNames.fontTitleXs} counter`}>
          {isFiltered ? `${filteredCount} of ${totalCount}` : totalCount}
        </span>
      </div>
      {isFiltered && (
        <Button text="Reset Filters" variant="secondary" size="s" className="reset-filters-button"></Button>
      )}
      <Search placeholder="Filter notebooks by name" className="notebooks-search" onSearch={onSearch}></Search>
    </div>
  );
};
