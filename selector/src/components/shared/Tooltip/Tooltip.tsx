import './Tooltip.scss';

import { ReactNode, useState } from 'react';

const sparkClassNames = {
  tooltipToggle: 'spark-tooltip-toggle',
  tooltip: 'spark-tooltip spark-tooltip-size-m spark-shadow',
  tooltipPlacementPrefix: 'spark-tooltip-',
  tooltipLabel: 'spark-tooltip-label',
  tooltipTip: 'spark-tooltip-tip',
};

type TooltipPlacement = 'top' | 'bottom' | 'right' | 'left' | 'bottom-end';

type TooltipProps = {
  content: string | JSX.Element;
  children?: ReactNode;
  placement?: TooltipPlacement;
  className?: string;
};

export const Tooltip = ({ content, children, placement = 'bottom', className = '' }: TooltipProps): JSX.Element => {
  const [isHovered, setIsHovered] = useState(false);

  const tooltipPlacementClassName =
    placement === 'bottom-end'
      ? `${sparkClassNames.tooltipPlacementPrefix}bottom ${sparkClassNames.tooltipPlacementPrefix}left`
      : `${sparkClassNames.tooltipPlacementPrefix}${placement}`;

  const tooltipClassNames = [sparkClassNames.tooltip, tooltipPlacementClassName].join(' ');
  return (
    <div
      className={`${sparkClassNames.tooltipToggle} ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {children}
      {isHovered && (
        <div className={tooltipClassNames}>
          <span className={sparkClassNames.tooltipLabel}>{content}</span>
          <span className={sparkClassNames.tooltipTip}></span>
        </div>
      )}
    </div>
  );
};
