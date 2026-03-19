import '../NotebookCard/NotebookCard.scss';

import GitHubIcon from '@assets/images/github.svg?react';
import OpenvinoLogo from '@assets/images/openvino-logo-colored.svg?react';

import { Button } from '@/components/shared/Button/Button';
import { Tag } from '@/components/shared/Tag/Tag';
import { IArchivedNotebookMetadata } from '@/shared/notebook-metadata';

const sparkClassNames = {
  card: 'spark-card spark-card-horizontal spark-card-border-normal',
  cardImage: 'spark-card-horizontal-bg-image spark-card-bg-fit-cover',
  cardTitle: 'spark-heading spark-font-100 spark-card-horizontal-title',
  fontCardDescription: 'spark-font-50',
  fontImagePlaceholder: 'spark-font-200',
  cardHorizontalLine: 'spark-card-horizontal-line',
};

type ArchivedNotebookCardProps = {
  item: IArchivedNotebookMetadata;
};

export const ArchivedNotebookCard = ({ item }: ArchivedNotebookCardProps): JSX.Element => {
  const { tasks, categories } = item.tags;
  const descriptionTags = [...categories, ...tasks];

  return (
    <div className={sparkClassNames.card}>
      <div className="card-wrapper">
        <div className="card-image-container">
          <div className="card-image-placeholder">
            <OpenvinoLogo></OpenvinoLogo>
            <span className={sparkClassNames.fontImagePlaceholder}>Notebooks</span>
          </div>
          {item.imageUrl && <img src={item.imageUrl} alt={item.title} className="card-image" />}
        </div>
        <div className="card-content">
          <h6 className={sparkClassNames.cardTitle}>
            <Tag text={`📦 ${item.lastBranch}`} theme="moss" variant="action"></Tag>
            <span>{item.title}</span>
          </h6>
          {descriptionTags.length > 0 && (
            <div className={`${sparkClassNames.fontCardDescription} card-description`}>
              {descriptionTags.join(' • ')}
            </div>
          )}
          <div className="card-footer">
            <div className={sparkClassNames.cardHorizontalLine}></div>
            <div className="card-actions">
              <Button
                as="link"
                variant="action"
                size="m"
                text="View on GitHub"
                icon={GitHubIcon}
                href={item.githubUrl}
              ></Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
