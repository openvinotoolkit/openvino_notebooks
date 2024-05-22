import './NotebookCard.scss';

import BinderIcon from '@assets/images/binder.svg?react';
import ColabIcon from '@assets/images/colab.svg?react';
import GitHubIcon from '@assets/images/github.svg?react';
import OpenvinoLogo from '@assets/images/openvino-logo-colored.svg?react';
import React, { CSSProperties, useRef, useState } from 'react';

import { Button } from '@/components/shared/Button/Button';
import { Tag } from '@/components/shared/Tag/Tag';
import { isEmbedded } from '@/shared/iframe-detector';
import { INotebookMetadata } from '@/shared/notebook-metadata';
import { CATEGORIES } from '@/shared/notebook-tags';
import { NotebookItem } from '@/shared/notebooks.service';

import { StatusTable } from './StatusTable/StatusTable';

const htmlToText = (value: string): string => {
  const div = document.createElement('div');
  div.innerHTML = value;
  return div.textContent || value;
};

const openNotebookInDocs = ({ links }: INotebookMetadata) => {
  if (!links.docs) {
    return;
  }
  window.open(links.docs, isEmbedded ? '_parent' : '_blank');
};

const getPointerLeftOffset = (buttonRef: React.RefObject<HTMLButtonElement>): string => {
  if (!buttonRef.current) {
    return '1rem';
  }
  const { offsetLeft, offsetWidth } = buttonRef.current;
  const pointerSize = 10;
  return `${offsetLeft + offsetWidth / 2 - pointerSize}px`;
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
  item: NotebookItem;
  showTasks?: boolean;
};

export const NotebookCard = ({ item, showTasks = true }: NotebookCardProps): JSX.Element => {
  const [isStatusVisible, showStatus] = useState(false);
  const statusButtonRef = useRef<HTMLButtonElement>(null);
  const { categories, tasks } = item.tags;
  const descriptionTags = [...categories.filter((v) => v !== CATEGORIES.AI_TRENDS), ...tasks];
  return (
    <div
      className={`${sparkClassNames.card} ${item.links.docs ? 'clickable' : ''}`}
      onClick={() => openNotebookInDocs(item)}
    >
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
                as="link"
                variant="action"
                size="m"
                text="View on GitHub"
                icon={GitHubIcon}
                href={item.links.github}
              ></Button>
              {item.links.colab && (
                <Button
                  as="link"
                  variant="primary"
                  size="m"
                  text="Open in Colab"
                  icon={ColabIcon}
                  href={item.links.colab}
                ></Button>
              )}
              {item.links.binder && (
                <Button
                  as="link"
                  variant="primary"
                  size="m"
                  text="Launch in Binder"
                  icon={BinderIcon}
                  href={item.links.binder}
                ></Button>
              )}
              {item.status && (
                <Button
                  ref={statusButtonRef}
                  as="button"
                  variant="secondary"
                  size="m"
                  text={`${isStatusVisible ? 'Hide' : 'Show'} Status`}
                  onClick={() => showStatus(!isStatusVisible)}
                ></Button>
              )}
            </div>
          </div>
        </div>
      </div>
      {isStatusVisible && (
        <div
          className="card-footer-panel"
          style={
            {
              '--pointer-left-offset': getPointerLeftOffset(statusButtonRef),
            } as CSSProperties
          }
        >
          <StatusTable status={item.status!} />
        </div>
      )}
    </div>
  );
};
