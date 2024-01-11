import './FiltersPanel.scss';

import { useState } from 'react';

import { FilterSection } from '@/components/shared/FilterSection/FilterSection';
import { Search } from '@/components/shared/Search/Search';
import { ITabItem, Tabs } from '@/components/shared/Tabs/Tabs';
import { CATEGORIES, TASKS_VALUES } from '@/models/notebook-tags';

// TODO Consider moving to models
interface IFilterGroup<T extends string = string> {
  title: string;
  group: T;
  tags: string[];
}

// TODO Consider moving to models
interface INotebookTags {
  categories: string[];
  tasks: string[];
  models: string[];
  libraries: string[];
  other: string[];
}

const initialTags: INotebookTags = {
  categories: [],
  tasks: [],
  models: [],
  libraries: [],
  other: [],
};

type FilterGroupKey = keyof INotebookTags;

const filterGroups: IFilterGroup<FilterGroupKey>[] = [
  {
    title: 'Categories',
    group: 'categories',
    tags: Object.values(CATEGORIES),
  },
  { title: 'Tasks', group: 'tasks', tags: TASKS_VALUES },
  { title: 'Libraries', group: 'libraries', tags: ['Tensorflow', 'PyTorch'] },
  { title: 'Other', group: 'other', tags: ['INT8'] },
];

export const FiltersPanel = (): JSX.Element => {
  const [selectedTags, setSelectedTags] = useState<INotebookTags>(initialTags);

  const handleTagClick = (tag: string, group: FilterGroupKey): void => {
    if (selectedTags[group].includes(tag)) {
      setSelectedTags({
        ...selectedTags,
        [group]: selectedTags[group].filter((v) => v !== tag),
      });
    } else {
      setSelectedTags({
        ...selectedTags,
        [group]: [...selectedTags[group], tag],
      });
    }
  };

  const tabItems: ITabItem[] = filterGroups.map(({ title, group, tags }) => ({
    key: group,
    title,
    badge: selectedTags[group].length,
    content: (
      <>
        <Search key={`search-${group}`} placeholder={`Filter ${title} by name`} className="filters-search"></Search>
        <FilterSection<FilterGroupKey>
          group={group}
          tags={tags}
          selectedTags={selectedTags[group]}
          onTagClick={(tag, group) => handleTagClick(tag, group!)}
        ></FilterSection>
      </>
    ),
  }));

  return (
    <section className="flex-col filters-panel">
      <Tabs items={tabItems}></Tabs>
    </section>
  );
};
