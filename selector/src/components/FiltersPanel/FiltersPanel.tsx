import './FiltersPanel.scss';

import { useContext, useState } from 'react';

import { FilterSection } from '@/components/shared/FilterSection/FilterSection';
import { Search } from '@/components/shared/Search/Search';
import { ITabItem, Tabs } from '@/components/shared/Tabs/Tabs';
import { INotebookMetadata } from '@/shared/notebook-metadata';
import { CATEGORIES, TASKS, TASKS_VALUES } from '@/shared/notebook-tags';
import { NotebooksContext } from '@/shared/notebooks-context';

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
];

const taskSectionTitlesMap: Record<keyof typeof TASKS, string> = {
  MULTIMODAL: 'Multimodal',
  CV: 'Computer Vision',
  NLP: 'Natural Language Processing',
  AUDIO: 'Audio',
};

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

  const tasksFilterSections = Object.entries(TASKS).map(([sectionKey, tagsMap]) => {
    const group = 'tasks';
    const title = taskSectionTitlesMap[sectionKey as keyof typeof TASKS];
    const filteredTags = filterTags(Object.values(tagsMap));
    if (!filteredTags.length) {
      return <></>;
    }
    return (
      <FilterSection<typeof group>
        key={`${group}-${sectionKey}`}
        title={title}
        group={group}
        tags={filteredTags}
        selectedTags={selectedTags[group]}
        onTagClick={(tag, group) => handleTagClick(tag, group!)}
      ></FilterSection>
    );
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
        ) : (
          <FilterSection<FilterGroupKey>
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
      <Tabs items={tabItems} onTabChange={() => setTagsSearch('')}></Tabs>
    </section>
  );
};
