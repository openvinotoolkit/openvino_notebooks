import './Dropdown.scss';

import ChevronIcon from '@assets/images/chevron.svg?react';
import { ForwardedRef, forwardRef, useEffect, useRef, useState } from 'react';

const useIsOpened = () => {
  const [isOpened, setIsOpened] = useState<boolean>(false);

  const ref = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) {
        setIsOpened(!isOpened);
      }
    };

    document.addEventListener('click', handleClickOutside, !isOpened);

    return () => {
      document.removeEventListener('click', handleClickOutside, !isOpened);
    };
  }, [isOpened]);

  return { ref, isOpened, setIsOpened };
};

const sparkClassNames = {
  dropdown: 'spark-dropdown spark-dropdown-primary spark-dropdown-size-m',
  dropdownButton:
    'spark-button spark-button-action spark-button-size-m spark-focus-visible spark-focus-visible-self spark-focus-visible-snap spark-dropdown-button spark-focus-visible spark-focus-visible-self spark-focus-visible-snap',
  dropdownButtonContent: 'spark-button-content',
  dropdownButtonLabel: 'spark-dropdown-button-label',
  dropdownButtonIcon: 'spark-icon spark-icon-regular spark-dropdown-arrow-icon',
  popover: 'spark-popover spark-shadow',
  dropdownListWrapper:
    'spark-scrollbar spark-scrollbar-y spark-focus-visible spark-focus-visible-self spark-focus-visible-snap spark-dropdown-list-box-scroll',
  dropdownList: 'spark-list spark-list-size-m spark-dropdown-list-box spark-dropdown-primary',
  dropdownListItem: 'spark-list-item',
  dropdownListActiveItem: 'spark-list-is-focused',
  dropdownListItemText: 'spark-list-item-text',
} as const;

type DropdownPopoverProps = {
  items: { text: string; onClick: () => void }[];
  direction: DropdownProps['direction'];
  selectedOption: string | null;
};

const DropdownPopover = forwardRef(function DropdownPopover(
  { items, selectedOption, direction = 'bottom' }: DropdownPopoverProps,
  ref: ForwardedRef<HTMLDivElement>
): JSX.Element {
  const directionClassName = `dropdown-popover-${direction}`;
  return (
    <div ref={ref} className={`${sparkClassNames.popover} ${directionClassName}`}>
      <div tabIndex={-1} className={sparkClassNames.dropdownListWrapper} role="group">
        <ul className={sparkClassNames.dropdownList} role="listbox" tabIndex={-1}>
          {items.map(({ text, onClick }, i) => (
            <li
              key={`dropdown-item-${i}-${text}`}
              className={`${sparkClassNames.dropdownListItem} ${
                text === selectedOption ? sparkClassNames.dropdownListActiveItem : ''
              }`}
              role="option"
              tabIndex={0}
              onClick={onClick}
            >
              <span className={sparkClassNames.dropdownListItemText}>{text}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
});

type DropdownProps = {
  options: string[];
  selectedOption: string | null;
  onSelect: (value: string) => void;
  placeholder?: string;
  selectedPrefix?: string;
  className?: string;
  direction?: 'bottom' | 'top';
};

export const Dropdown = ({
  options,
  selectedOption,
  onSelect,
  placeholder = 'Select an Option',
  selectedPrefix = '',
  className = '',
  direction = 'bottom',
}: DropdownProps): JSX.Element => {
  const { ref, isOpened, setIsOpened } = useIsOpened();

  const selectedOptionText = selectedPrefix ? `${selectedPrefix}: ${selectedOption}` : selectedOption;

  return (
    <>
      <div className={`${sparkClassNames.dropdown} ${className}`}>
        <button
          ref={ref}
          className={sparkClassNames.dropdownButton}
          type="button"
          aria-haspopup="listbox"
          aria-expanded={isOpened}
          onClick={() => setIsOpened(!isOpened)}
        >
          <span className={sparkClassNames.dropdownButtonContent}>
            <span className={sparkClassNames.dropdownButtonLabel}>
              {selectedOption ? selectedOptionText : placeholder}
            </span>
            <span aria-hidden="true" role="img" className={sparkClassNames.dropdownButtonIcon}>
              <ChevronIcon />
            </span>
          </span>
        </button>
        {isOpened && (
          <DropdownPopover
            direction={direction}
            selectedOption={selectedOption}
            items={options.map((option) => ({
              text: option,
              onClick: () => {
                onSelect(option);
                setIsOpened(false);
              },
            }))}
          />
        )}
      </div>
    </>
  );
};
