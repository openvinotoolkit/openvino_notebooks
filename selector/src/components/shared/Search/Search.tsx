import './Search.scss';

import CrossIcon from '@assets/images/cross.svg?react';
import SearchIcon from '@assets/images/search.svg?react';

const sparkClassNames = {
  textFieldContainer: 'spark-text-field-container',
  fieldTextWrapper: 'spark-fieldtext-wrapper',
  textField:
    'spark-text-field spark-text-field-outline spark-text-field-size-m spark-text-field-start-slot-1x spark-text-field-end-slot-1x',
  textFieldStartSlot: 'spark-text-field-start-slot',
  input: 'spark-input spark-input-outline spark-input-size-m spark-focus spark-focus-within spark-focus-snap',
  textFieldEndSlot: 'spark-text-field-end-slot',
  iconButton: 'spark-button spark-button-ghost',
  icon: 'spark-icon',
};

type SearchProps = {
  placeholder?: string;
  className?: string;
  value?: string;
  search?: (value: string) => void;
};

export const Search = ({ placeholder, className = '', value = '', search }: SearchProps): JSX.Element => {
  return (
    <div className={`${sparkClassNames.textFieldContainer} ${className}`}>
      <div className={sparkClassNames.fieldTextWrapper}>
        <div className={sparkClassNames.textField}>
          <div className={sparkClassNames.textFieldStartSlot}>
            <SearchIcon className={sparkClassNames.icon}></SearchIcon>
          </div>
          <input
            type="text"
            className={sparkClassNames.input}
            placeholder={placeholder}
            inputMode="text"
            value={value}
            onChange={(event) => search?.(event.target.value)}
          />
          <div className={sparkClassNames.textFieldEndSlot}>
            {value && (
              <button className={sparkClassNames.iconButton} onClick={() => search?.('')}>
                <CrossIcon className={sparkClassNames.icon} width="10"></CrossIcon>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
