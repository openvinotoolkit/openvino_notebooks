import './Badge.scss';

const sparkClassNames = {
  badge: 'spark-badge spark-badge-variant-info spark-badge-shape-circle',
  badgeSizePrefix: 'spark-badge-text-size-',
  badgeText: 'spark-badge-text',
};

type BageSize = 's' | 'xs';

type BadgeProps = {
  text: string;
  size?: BageSize;
};

export const Badge = (props: BadgeProps): JSX.Element => {
  const { text, size = 's' } = props;
  const sizeClassName = `${sparkClassNames.badgeSizePrefix}${size}`;
  return (
    <span className={`${sparkClassNames.badge} ${sizeClassName}`}>
      <span className={sparkClassNames.badgeText}>{text}</span>
    </span>
  );
};
