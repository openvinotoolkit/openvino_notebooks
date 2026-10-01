import './NotebookCard.scss';

import BinderIcon from '@assets/images/binder.svg?react';
import ColabIcon from '@assets/images/colab.svg?react';
import GitHubIcon from '@assets/images/github.svg?react';
import LinkIcon from '@assets/images/link.svg?react';
import OpenvinoLogo from '@assets/images/openvino-logo-colored.svg?react';
import { JSX, useContext, useLayoutEffect, useRef, useState } from 'react';

import { Button } from '@/components/shared/Button/Button';
import { Tag } from '@/components/shared/Tag/Tag';
import { Tooltip } from '@/components/shared/Tooltip/Tooltip';
import { analytics } from '@/shared/analytics/analytics';
import { copyToClipboard } from '@/shared/copy';
import { isEmbedded } from '@/shared/iframe-detector';
import { INotebookMetadata } from '@/shared/notebook-metadata';
import { CATEGORIES } from '@/shared/notebook-tags';
import { NotebookItem } from '@/shared/notebooks.service';
import { NotebooksContext } from '@/shared/notebooks-context';
import { getUrlParamsWithSearch } from '@/shared/selectorUrlPersist';

import { StatusTable } from './StatusTable/StatusTable';

const htmlToText = (value: string): string => {
  const div = document.createElement('div');
  div.innerHTML = value;
  return div.textContent || value;
};

const openNotebookInDocs = ({ links, path }: INotebookMetadata) => {
  if (!links.docs) {
    return;
  }
  analytics.sendNavigateEvent(path, links.docs);
  window.open(links.docs, isEmbedded ? '_parent' : '_blank');
};

const copyNotebookShareUrl = ({ title }: INotebookMetadata): void => {
  const shareUrl = new URL(window.location.toString());
  shareUrl.search = getUrlParamsWithSearch(title).toString();
  void copyToClipboard(shareUrl.toString());
};

const sparkClassNames = {
  card: 'spark-card spark-card-horizontal spark-card-border-normal',
  cardImage: 'spark-card-horizontal-bg-image spark-card-bg-fit-cover',
  cardTitle: 'spark-heading spark-font-100 spark-card-horizontal-title',
  fontCardDescription: 'spark-font-50',
  fontImagePlaceholder: 'spark-font-200',
  cardHorizontalLine: 'spark-card-horizontal-line',
};

const MAX_MATCHED_MODELS = 3;

/**
 * Returns the models matching the search, preferring the short name over the full `org/name` id.
 */
const getMatchedModels = (models: string[], searchValue: string): string[] => {
  const search = searchValue.trim().toLowerCase();
  if (!search) {
    return [];
  }
  const matched = models.filter((model) => model.toLowerCase().includes(search));
  const shortMatched = new Set(matched.filter((model) => !model.includes('/')));
  return matched.filter((model) => !model.includes('/') || !shortMatched.has(model.split('/').pop()!));
};

type NotebookCardProps = {
  item: NotebookItem;
  showTasks?: boolean;
};

export const NotebookCard = ({ item, showTasks = true }: NotebookCardProps): JSX.Element => {
  const [isStatusVisible, showStatus] = useState(false);
  const [isLinkCopied, setLinkCopied] = useState(false);
  const [areModelsExpanded, setModelsExpanded] = useState(false);
  const statusButtonRef = useRef<HTMLButtonElement>(null);
  const statusPanelRef = useRef<HTMLDivElement>(null);
  const { searchValue } = useContext(NotebooksContext);
  const { categories, tasks } = item.tags;
  const descriptionTags = [...categories.filter((v) => v !== CATEGORIES.AI_TRENDS), ...tasks];
  const matchedModels = getMatchedModels(item.models, searchValue);
  const titleMatchesSearch = htmlToText(item.title).toLowerCase().includes(searchValue.trim().toLowerCase());
  const shownModels = areModelsExpanded ? matchedModels : matchedModels.slice(0, MAX_MATCHED_MODELS);
  const hiddenModelsCount = matchedModels.length - shownModels.length;

  // Measure after each commit so button labels and the action layout are up to date.
  useLayoutEffect(() => {
    const button = statusButtonRef.current;
    const panel = statusPanelRef.current;
    if (!button || !panel) {
      return;
    }
    const pointerSize = 10;
    panel.style.setProperty('--pointer-left-offset', `${button.offsetLeft + button.offsetWidth / 2 - pointerSize}px`);
  });

  return (
    <div className={sparkClassNames.card}>
      <div className={`card-wrapper ${item.links.docs ? 'clickable' : ''}`} onClick={() => openNotebookInDocs(item)}>
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
          {!titleMatchesSearch && shownModels.length > 0 && (
            <div className="card-matched-models">
              <span className={sparkClassNames.fontCardDescription}>Matched models:</span>
              {shownModels.map((model) => (
                <Tag key={model} text={model} theme="cobalt" variant="secondary"></Tag>
              ))}
              {(hiddenModelsCount > 0 || areModelsExpanded) && (
                <button
                  type="button"
                  className="card-matched-models-toggle"
                  onClick={(e) => {
                    e.stopPropagation();
                    setModelsExpanded(!areModelsExpanded);
                  }}
                >
                  {areModelsExpanded ? 'Show less' : `+${hiddenModelsCount} more`}
                </button>
              )}
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
                onClick={() => {
                  analytics.sendNavigateEvent(item.path, item.links.github);
                }}
              ></Button>
              {item.links.colab && (
                <Button
                  as="link"
                  variant="primary"
                  size="m"
                  text="Open in Colab"
                  icon={ColabIcon}
                  href={item.links.colab}
                  onClick={() => {
                    analytics.sendNavigateEvent(item.path, item.links.colab!);
                  }}
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
                  onClick={() => {
                    analytics.sendNavigateEvent(item.path, item.links.binder!);
                  }}
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
              <Tooltip content={isLinkCopied ? 'Copied' : 'Copy link to clipboard'}>
                <Button
                  as="button"
                  variant="secondary"
                  size="m"
                  icon={LinkIcon}
                  onClick={() => {
                    copyNotebookShareUrl(item);
                    analytics.sendCopyLinkEvent(item.path);
                    setLinkCopied(true);
                    setTimeout(() => setLinkCopied(false), 1000);
                  }}
                ></Button>
              </Tooltip>
            </div>
          </div>
        </div>
      </div>
      {isStatusVisible && (
        <div ref={statusPanelRef} className="card-footer-panel">
          <StatusTable status={item.status!} />
        </div>
      )}
    </div>
  );
};
