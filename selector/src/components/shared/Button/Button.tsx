import './Button.scss';

import { AnchorHTMLAttributes, ButtonHTMLAttributes, forwardRef, FunctionComponent, SVGProps } from 'react';

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

type AsElementProps =
  | (ButtonHTMLAttributes<HTMLButtonElement> & { as?: 'button' })
  | (AnchorHTMLAttributes<HTMLAnchorElement> & { as?: 'link' });

type ButtonProps = {
  text?: string;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  value?: string;
  onClick?: () => void;
  icon?: FunctionComponent<SVGProps<SVGSVGElement>>;
  className?: string;
} & AsElementProps;

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { text, as = 'button', variant = 'primary', size = 'm', disabled = false, onClick, icon, className, ...props },
  ref
): JSX.Element {
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

  const buttonContent = (
    <>
      {icon && <span className={sparkClassNames.buttonStartSlot}>{icon({ className: 'button-icon' })}</span>}
      <span className={sparkClassNames.buttonContent}>{text}</span>
    </>
  );

  if (as === 'link') {
    return (
      <a
        className={classNames.join(' ')}
        href={(props as AnchorHTMLAttributes<HTMLAnchorElement>).href}
        target="_blank"
        rel="noreferrer"
        aria-label={text}
        onClick={(e) => {
          e.stopPropagation();
          onClick?.();
        }}
      >
        {buttonContent}
      </a>
    );
  }

  return (
    <button
      className={classNames.join(' ')}
      type="button"
      role="radio"
      onClick={(e) => {
        e.stopPropagation();
        onClick?.();
      }}
      aria-label={text}
      ref={ref}
    >
      {buttonContent}
    </button>
  );
});
