import './FiltersPanel.scss';

import { useContext, useState } from 'react';

import CrossIcon from '@/assets/images/cross.svg?react';
import { FilterSection } from '@/components/shared/FilterSection/FilterSection';
import { Search } from '@/components/shared/Search/Search';
import { ITabItem, Tabs } from '@/components/shared/Tabs/Tabs';
import { INotebookMetadata } from '@/shared/notebook-metadata';
import { CATEGORIES, LIBRARIES, LIBRARIES_VALUES, TASKS, TASKS_VALUES } from '@/shared/notebook-tags';
import { NotebooksContext } from '@/shared/notebooks-context';

import { Button } from '../shared/Button/Button';
import { closeFiltersPanel } from './filters-panel-handlers';

interface IFilterGroup<T extends string = string> {
  title: string;
  group: T;
  tags: string[];
}

type FilterGroupKey = keyof INotebookMetadata['tags'];

const filterGroups: IFilterGroup<FilterGroupKey>[] = [
  {
    title: 'Categories',
    group: 'categories',
    tags: Object.values(CATEGORIES),
  },
  { title: 'AI Tasks', group: 'tasks', tags: TASKS_VALUES },
  { title: 'Ecosystem', group: 'libraries', tags: LIBRARIES_VALUES },
];

const tasksSectionsTitlesMap: Record<keyof typeof TASKS, string> = {
  MULTIMODAL: 'Multimodal',
  CV: 'Computer Vision',
  NLP: 'Natural Language Processing',
  AUDIO: 'Audio',
  OTHER: 'Other',
};

const librariesSectionsTitlesMap: Record<keyof typeof LIBRARIES, string> = {
  OPENVINO: 'OpenVINO',
  OTHER: 'Other Tools',
};

function getTagsFilterSections<T extends Record<string, Record<string, string>>>({
  group,
  tagsMap,
  titlesMap,
  selectedTags,
  filterTags,
  handleTagClick,
}: {
  group: keyof INotebookMetadata['tags'];
  tagsMap: T;
  titlesMap: Record<keyof T, string>;
  selectedTags: INotebookMetadata['tags'];
  filterTags: (tags: string[]) => string[];
  handleTagClick: (tag: string, group: FilterGroupKey) => void;
}): JSX.Element[] {
  return Object.entries(tagsMap).map(([sectionKey, tagsMap]) => {
    const title = titlesMap[sectionKey];
    const filteredTags = filterTags(Object.values(tagsMap));
    if (!filteredTags.length) {
      return <></>;
    }
    return (
      <FilterSection<typeof group>
        key={`filter-section-${group}-${sectionKey}`}
        title={title}
        group={group}
        tags={filteredTags}
        selectedTags={selectedTags[group]}
        onTagClick={(tag, group) => handleTagClick(tag, group!)}
      ></FilterSection>
    );
  });
}

export const FiltersPanel = (): JSX.Element => {
  const { selectedTags, setSelectedTags } = useContext(NotebooksContext);

  const [tagsSearch, setTagsSearch] = useState('');

  const filterTags = (tags: string[]): string[] =>
    tags.filter((tag) => tag.toLowerCase().includes(tagsSearch.toLowerCase()));

  const handleTagClick = (tag: string, group: FilterGroupKey): void => {
    const tags = selectedTags[group] as string[];
    if (tags.includes(tag)) {
      setSelectedTags({
        ...selectedTags,
        [group]: tags.filter((v) => v !== tag),
      });
    } else {
      setSelectedTags({
        ...selectedTags,
        [group]: [tag],
      });
    }
  };

  const tasksFilterSections = getTagsFilterSections<typeof TASKS>({
    group: 'tasks',
    tagsMap: TASKS,
    titlesMap: tasksSectionsTitlesMap,
    selectedTags,
    filterTags,
    handleTagClick,
  });

  const librariesFilterSections = getTagsFilterSections<typeof LIBRARIES>({
    group: 'libraries',
    tagsMap: LIBRARIES,
    titlesMap: librariesSectionsTitlesMap,
    selectedTags,
    filterTags,
    handleTagClick,
  });

  const tabItems: ITabItem[] = filterGroups.map(({ title, group, tags }) => ({
    key: group,
    title,
    badge: selectedTags[group].length,
    content: (
      <>
        <Search
          key={`search-${group}`}
          placeholder={`Filter ${title} by name`}
          className="filters-search"
          value={tagsSearch}
          search={setTagsSearch}
        ></Search>
        {group === 'tasks' ? (
          tasksFilterSections
        ) : group === 'libraries' ? (
          librariesFilterSections
        ) : (
          <FilterSection<FilterGroupKey>
            key={`filter-section-${group}`}
            group={group}
            tags={filterTags(tags)}
            selectedTags={selectedTags[group]}
            onTagClick={(tag, group) => handleTagClick(tag, group!)}
          ></FilterSection>
        )}
      </>
    ),
  }));

  return (
    <section className="flex-col filters-panel">
      <div className="lg-hidden filters-panel-header">
        <span>Edit Notebooks filters</span>
        <span onClick={closeFiltersPanel} className="close-icon">
          <CrossIcon />
        </span>
      </div>
      <Tabs items={tabItems} onTabChange={() => setTagsSearch('')}></Tabs>
      <div className="lg-hidden filters-panel-footer">
        <Button
          onClick={closeFiltersPanel}
          text="Apply filters"
          variant="action"
          size="l"
          className="apply-filters-button"
        ></Button>
      </div>
    </section>
  );
};
