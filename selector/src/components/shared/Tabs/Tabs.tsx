import './Tabs.scss';

import { Badge } from '@components/shared/Badge/Badge';
import { useState } from 'react';

const sparkClassNames = {
  tab: 'spark-button spark-button-ghost spark-button-size-m spark-focus-visible spark-focus-visible-self spark-focus-visible-snap spark-tabs-tab',
  tabActive: 'spark-tabs-active',
  tabContent: 'spark-button-content',
  tabs: 'spark-tabs spark-tabs-size-m spark-tabs-ghost',
};

type TabProps = {
  title: string;
  active?: boolean;
  badge?: string;
  onClick?: () => void;
};

const Tab = ({ title, active, badge, onClick }: TabProps): JSX.Element => (
  <button
    className={`${sparkClassNames.tab} ${active ? sparkClassNames.tabActive : ''}`}
    type="button"
    tabIndex={-1}
    role="tab"
    aria-selected="true"
    onClick={onClick}
  >
    <span className={sparkClassNames.tabContent}>{title}</span>
    {badge && (
      <span className="superscript-badge">
        <Badge text={badge} size="xs" />
      </span>
    )}
  </button>
);

export interface ITabItem {
  title: string;
  content: JSX.Element;
}

type TabsProps = {
  items: ITabItem[];
};

export const Tabs = ({ items }: TabsProps): JSX.Element => {
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);

  return (
    <div className="tabs-container">
      <nav className={sparkClassNames.tabs} aria-label="Tabs" role="tablist" aria-orientation="horizontal">
        {items.map(({ title }, i) => (
          <Tab key={`tab-${i}`} title={title} active={i === selectedTabIndex} onClick={() => setSelectedTabIndex(i)} />
        ))}
      </nav>
      <div className="tab-content">{items[selectedTabIndex].content}</div>
    </div>
  );
};
