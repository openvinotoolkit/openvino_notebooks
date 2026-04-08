import './ContentSectionHeader.scss';

import { useContext, useEffect } from 'react';

import { openFiltersPanel } from '@/components/FiltersPanel/filters-panel-handlers';
import { Button } from '@/components/shared/Button/Button';
import { Dropdown } from '@/components/shared/Dropdown/Dropdown';
import { Search } from '@/components/shared/Search/Search';
import { analytics } from '@/shared/analytics/analytics';
import { SORT_OPTIONS, SortValues } from '@/shared/notebooks.service';
import { NotebooksContext, ViewMode } from '@/shared/notebooks-context';

const sparkClassNames = {
  fontTitleXs: 'spark-font-200',
  tab: 'spark-button spark-button-ghost spark-button-size-m spark-focus-visible spark-focus-visible-self spark-focus-visible-snap spark-tabs-tab',
  tabActive: 'spark-tabs-active',
  tabContent: 'spark-button-content',
  tabs: 'spark-tabs spark-tabs-size-m spark-tabs-ghost',
};

type ContentSectionHeaderProps = {
  totalActiveCount: number;
  filteredCount: number;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  totalArchivedCount: number;
};

export const ContentSectionHeader = ({
  totalActiveCount,
  filteredCount,
  viewMode,
  onViewModeChange,
  totalArchivedCount,
}: ContentSectionHeaderProps): JSX.Element => {
  const { searchValue, setSearchValue, resetFilters, sort, setSort } = useContext(NotebooksContext);

  const totalCount = viewMode === 'active' ? totalActiveCount : totalArchivedCount;
  const isFiltered = filteredCount !== totalCount;

  // Send search event to analytics with debouncing
  useEffect(() => {
    const sendSearchEventTimeout = setTimeout(() => {
      if (searchValue) {
        analytics.sendSearchEvent(searchValue);
      }
    }, 2000);

    return () => clearTimeout(sendSearchEventTimeout);
  }, [searchValue]);

  return (
    <div className="content-section-header">
      <div className="title-container">
        <nav className={sparkClassNames.tabs} aria-label="View mode" role="tablist">
          <button
            className={`${sparkClassNames.tab} ${viewMode === 'active' ? sparkClassNames.tabActive : ''}`}
            type="button"
            role="tab"
            aria-selected={viewMode === 'active'}
            onClick={() => onViewModeChange('active')}
          >
            <span className={sparkClassNames.tabContent}>
              Notebooks ({isFiltered && viewMode === 'active' ? `${filteredCount} of ${totalActiveCount}` : totalActiveCount})
            </span>
          </button>
          <button
            className={`${sparkClassNames.tab} ${viewMode === 'archived' ? sparkClassNames.tabActive : ''}`}
            type="button"
            role="tab"
            aria-selected={viewMode === 'archived'}
            onClick={() => onViewModeChange('archived')}
          >
            <span className={sparkClassNames.tabContent}>
              Archived ({isFiltered && viewMode === 'archived' ? `${filteredCount} of ${totalArchivedCount}` : totalArchivedCount})
            </span>
          </button>
        </nav>
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
        {viewMode === 'active' && (
          <Button
            text={isFiltered ? 'Edit Filters' : 'Add Filters'}
            variant="secondary"
            size="m"
            className="lg-hidden edit-filters-button"
            onClick={openFiltersPanel}
          ></Button>
        )}

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
