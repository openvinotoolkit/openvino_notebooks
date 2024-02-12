import './NotebookCard.scss';

import BinderIcon from '@assets/images/binder.svg?react';
import ColabIcon from '@assets/images/colab.svg?react';
import GitHubIcon from '@assets/images/github.svg?react';
import OpenvinoLogo from '@assets/images/openvino-logo-colored.svg?react';

import { Button } from '@/components/shared/Button/Button';
import { Tag } from '@/components/shared/Tag/Tag';
import { INotebookMetadata } from '@/shared/notebook-metadata';
import { CATEGORIES } from '@/shared/notebook-tags';

const htmlToText = (value: string): string => {
  const div = document.createElement('div');
  div.innerHTML = value;
  return div.textContent || value;
};

const openLink = (url: string) => window.open(url, '_blank');

const openNotebookInDocs = ({ path }: INotebookMetadata) => {
  const fileNameRegexp = /\/(?<fileName>.+)\.ipynb/g;
  const notebookFileName = fileNameRegexp.exec(path)?.groups?.fileName;
  const url = `https://docs.openvino.ai/2023.3/notebooks/${notebookFileName}-with-output.html`;
  const isEmbedded = window !== window.parent;
  // TODO Disable opening docs page for ignored notebooks (and possibly for new not released notebooks)
  window.open(url, isEmbedded ? '_self' : '_blank');
};

const sparkClassNames = {
  card: 'spark-card spark-card-horizontal spark-card-border-normal',
  cardImage: 'spark-card-horizontal-bg-image spark-card-bg-fit-cover',
  cardTitle: 'spark-heading spark-font-100 spark-card-horizontal-title',
  fontCardDescription: 'spark-font-50',
  fontImagePlaceholder: 'spark-font-200',
  cardHorizontalLine: 'spark-card-horizontal-line',
};

type NotebookCardProps = {
  item: INotebookMetadata;
  showTasks?: boolean;
};

export const NotebookCard = ({ item, showTasks = true }: NotebookCardProps): JSX.Element => {
  const { categories, tasks } = item.tags;
  const descriptionTags = [...categories.filter((v) => v !== CATEGORIES.AI_TRENDS), ...tasks];
  return (
    <div className={sparkClassNames.card} onClick={() => openNotebookInDocs(item)}>
      <div className="card-wrapper">
        <div className="card-image-container">
          <div className="card-image-placeholder">
            <OpenvinoLogo></OpenvinoLogo>
            <span className={sparkClassNames.fontImagePlaceholder}>Notebooks</span>
          </div>
          {item.imageUrl && <img src={item.imageUrl} className="card-image" />}
        </div>
        <div className="card-content">
          <h6 className={sparkClassNames.cardTitle}>
            {item.tags.categories.includes(CATEGORIES.AI_TRENDS) && (
              <Tag text="🚀 AI Trends" theme="daisy-tint1" variant="action"></Tag>
            )}
            <span>{htmlToText(item.title)}</span>
          </h6>
          {showTasks && (
            <div className={`${sparkClassNames.fontCardDescription} card-description`}>
              {descriptionTags.join(' • ')}
            </div>
          )}
          <div className="card-footer">
            <div className={sparkClassNames.cardHorizontalLine}></div>
            <div className="card-actions">
              <Button
                text="View on GitHub"
                variant="action"
                icon={GitHubIcon}
                size="m"
                onClick={() => openLink(item.links.github)}
              ></Button>
              {item.links.colab && (
                <Button
                  text="Open in Colab"
                  variant="primary"
                  icon={ColabIcon}
                  size="m"
                  onClick={() => openLink(item.links.colab!)}
                ></Button>
              )}
              {item.links.binder && (
                <Button
                  text="Launch in Binder"
                  variant="primary"
                  icon={BinderIcon}
                  size="m"
                  onClick={() => openLink(item.links.binder!)}
                ></Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
