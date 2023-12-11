import './FiltersPanel.scss';

import { useState } from 'react';

import { FilterSection } from '../shared/FilterSection/FilterSection';
import { ITabItem, Tabs } from '../shared/Tabs/Tabs';

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
    tags: ['AI Trends', 'First Steps', 'Convert & Optimize', 'Model Demos', 'Model Training', 'Live Demos'],
  },
  { title: 'Tasks', group: 'tasks', tags: ['Multimodal', 'Computer Vision', 'Natural Language Processing', 'Audio'] },
  { title: 'Models', group: 'models', tags: ['ControlNet', 'MobileNet'] },
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
      <FilterSection<FilterGroupKey>
        group={group}
        tags={tags}
        selectedTags={selectedTags[group]}
        onTagClick={(tag, group) => handleTagClick(tag, group!)}
      ></FilterSection>
    ),
  }));

  return (
    <section className="flex-col filters-panel">
      <Tabs items={tabItems}></Tabs>
    </section>
  );
};
