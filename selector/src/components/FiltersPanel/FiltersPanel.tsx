import './FiltersPanel.scss';

import { ITabItem, Tabs } from '../shared/Tabs/Tabs';

// TODO Consider moving to models
interface IFilterGroup {
  title: string;
  tags: string[];
}

const filterGroups: IFilterGroup[] = [
  {
    title: 'Categories',
    tags: ['AI Trends', 'First Steps', 'Convert & Optimize', 'Model Demos', 'Model Training', 'Live Demos'],
  },
  { title: 'Tasks', tags: ['Multimodal', 'Computer Vision', 'Natural Language Processing', 'Audio'] },
];

export const FiltersPanel = (): JSX.Element => {
  const tabItems: ITabItem[] = filterGroups.map(({ title, tags }) => ({
    key: title,
    title,
    content: (
      <ul>
        {tags.map((v) => (
          <li key={v}>{v}</li>
        ))}
      </ul>
    ),
  }));

  return (
    <section className="flex-col filters-panel">
      <Tabs items={tabItems}></Tabs>
    </section>
  );
};
