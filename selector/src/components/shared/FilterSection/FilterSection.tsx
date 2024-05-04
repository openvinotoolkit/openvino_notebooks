import './FilterSection.scss';

import { Button } from '../Button/Button';

const sparkClassNames = {
  fontSubtitleM: 'spark-font-100',
};

type FilterSectionProps<T extends string = string> = {
  group?: T;
  title?: string;
  tags: string[];
  selectedTags: string[];
  onTagClick?: (tag: string, group: T | undefined) => void;
};

export const FilterSection = <T extends string = string>({
  group,
  title,
  tags,
  selectedTags = [],
  onTagClick,
}: FilterSectionProps<T>): JSX.Element => {
  return (
    <div className="filter-section">
      {title && <span className={`filter-section-title ${sparkClassNames.fontSubtitleM}`}>{title}</span>}
      <div className="filter-section-content">
        {tags.map((tag) => (
          <Button
            text={tag}
            key={group ? `${group}-${tag}` : tag}
            variant={selectedTags.includes(tag) ? 'action' : 'primary'}
            size="m"
            className="filter-tag"
            onClick={() => onTagClick?.(tag, group)}
          ></Button>
        ))}
      </div>
    </div>
  );
};
