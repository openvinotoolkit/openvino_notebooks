import './Button.scss';

import { FunctionComponent, SVGProps } from 'react';

const sparkClassNames = {
  button: 'spark-button spark-focus-visible spark-focus-visible-self spark-focus-visible-snap',
  buttonSizePrefix: 'spark-button-size-',
  buttonVariantPrefix: 'spark-button-',
  disabledButton: 'spark-button-disabled',
  buttonStartSlot: 'spark-button-start-slot',
  buttonContent: 'spark-button-content',
  buttonOnly: 'spark-button-icon-only',
};

type ButtonVariant = 'action' | 'primary' | 'secondary' | 'ghost';

type ButtonSize = 'l' | 'm' | 's';

type ButtonProps = {
  text?: string;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  value?: string;
  onClick?: () => void;
  icon?: FunctionComponent<SVGProps<SVGSVGElement>>;
  className?: string;
};

export const Button = ({
  text,
  variant = 'primary',
  size = 'm',
  disabled = false,
  onClick,
  icon,
  className,
}: ButtonProps): JSX.Element => {
  const sizeClassName = `${sparkClassNames.buttonSizePrefix}${size}`;
  const variantClassName = `${sparkClassNames.buttonVariantPrefix}${variant}`;
  const classNames = [sparkClassNames.button, sizeClassName, variantClassName];

  if (disabled) {
    classNames.push(sparkClassNames.disabledButton);
  }

  if (!text && icon) {
    classNames.push(sparkClassNames.buttonOnly);
  }

  if (className) {
    classNames.push(className);
  }

  return (
    <button className={classNames.join(' ')} type="button" role="radio" onClick={() => onClick?.()} aria-label={text}>
      {icon && <span className={sparkClassNames.buttonStartSlot}>{icon({ className: 'button-icon' })}</span>}
      <span className={sparkClassNames.buttonContent}>{text}</span>
    </button>
  );
};
