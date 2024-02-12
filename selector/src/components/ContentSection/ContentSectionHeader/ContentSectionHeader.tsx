import './ContentSectionHeader.scss';

import { useContext } from 'react';

import { openFiltersPanel } from '@/components/FiltersPanel/filters-panel-handlers';
import { Button } from '@/components/shared/Button/Button';
import { Dropdown } from '@/components/shared/Dropdown/Dropdown';
import { Search } from '@/components/shared/Search/Search';
import { SORT_OPTIONS, SortValues } from '@/shared/notebooks.service';
import { NotebooksContext } from '@/shared/notebooks-context';

const sparkClassNames = {
  fontTitleXs: 'spark-font-200',
};

type ContentSectionHeaderProps = {
  totalCount: number;
  filteredCount: number;
};

export const ContentSectionHeader = ({ totalCount, filteredCount }: ContentSectionHeaderProps): JSX.Element => {
  const { searchValue, setSearchValue, resetFilters, sort, setSort } = useContext(NotebooksContext);

  const isFiltered = filteredCount !== totalCount;

  return (
    <div className="content-section-header">
      <div className="title-container">
        <h1 className={`${sparkClassNames.fontTitleXs} title`}>Notebooks</h1>
        <span className={`${sparkClassNames.fontTitleXs} counter`}>
          {isFiltered ? `${filteredCount} of ${totalCount}` : totalCount}
        </span>
        {isFiltered && (
          <Button
            text="Reset Filters"
            variant="secondary"
            size="s"
            className="reset-filters-button"
            onClick={resetFilters}
          ></Button>
        )}
      </div>
      <div className="content-section-header-actions">
        <Button
          text={isFiltered ? 'Edit Filters' : 'Add Filters'}
          variant="secondary"
          size="m"
          className="lg-hidden edit-filters-button"
          onClick={openFiltersPanel}
        ></Button>

        <Search
          placeholder="Filter notebooks by name"
          className="notebooks-search"
          search={setSearchValue}
          value={searchValue}
        ></Search>

        <Dropdown
          className="notebooks-sort"
          options={Object.values(SORT_OPTIONS)}
          selectedOption={sort}
          selectedPrefix="Sort"
          onSelect={(option) => setSort(option as SortValues)}
        ></Dropdown>
      </div>
    </div>
  );
};
