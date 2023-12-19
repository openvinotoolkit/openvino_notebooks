import './NotebookCard.scss';

import BinderIcon from '@assets/images/binder.svg?react';
import ColabIcon from '@assets/images/colab.svg?react';
import GitHubIcon from '@assets/images/github.svg?react';
import OpenvinoLogo from '@assets/images/openvino-logo-colored.svg?react';

import { Button } from '@/components/shared/Button/Button';
import { INotebookMetadata } from '@/models/notebook';

const openLink = (url: string) => window.open(url, '_blank');

const sparkClassNames = {
  card: 'spark-card spark-card-horizontal spark-card-border-normal',
  cardImage: 'spark-card-horizontal-bg-image spark-card-bg-fit-cover',
  cardTitle: 'spark-heading spark-font-100 spark-card-horizontal-title',
  fontCardDescription: 'spark-font-75',
  fontImagePlaceholder: 'spark-font-300',
  cardHorizontalLine: 'spark-card-horizontal-line',
};

type NotebookCardProps = {
  item: INotebookMetadata;
};

export const NotebookCard = ({ item }: NotebookCardProps): JSX.Element => {
  return (
    <div className={sparkClassNames.card}>
      <div className="card-wrapper">
        <div className="card-image-container">
          <div className="card-image-placeholder">
            <OpenvinoLogo></OpenvinoLogo>
            <span className={sparkClassNames.fontImagePlaceholder}>Notebooks</span>
          </div>
          {item.imageUrl && <img src={item.imageUrl} className="card-image" />}
        </div>
        <div className="card-content">
          <h6 className={sparkClassNames.cardTitle}>{item.title}</h6>
          <div className="card-description">
            <span className={sparkClassNames.fontCardDescription}>{item.description}</span>
          </div>
          <div className="card-footer">
            <div className={sparkClassNames.cardHorizontalLine}></div>
            <div className="card-actions">
              <Button
                text="View on GitHub"
                variant="action"
                icon={GitHubIcon}
                onClick={() => openLink(item.links.github)}
              ></Button>
              {item.links.colab && (
                <Button
                  text="Open in Colab"
                  variant="primary"
                  icon={ColabIcon}
                  onClick={() => openLink(item.links.colab!)}
                ></Button>
              )}
              {item.links.binder && (
                <Button
                  text="Launch in Binder"
                  variant="primary"
                  icon={BinderIcon}
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
