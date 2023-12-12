import './Search.scss';

import CrossIcon from '@assets/images/cross.svg?react';
import SearchIcon from '@assets/images/search.svg?react';
import { useState } from 'react';

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
};

export const Search = ({ placeholder, className = '' }: SearchProps): JSX.Element => {
  const [searchValue, setSearchValue] = useState('');

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
            value={searchValue}
            onChange={(event) => setSearchValue(event.target.value)}
          />
          <div className={sparkClassNames.textFieldEndSlot}>
            {searchValue && (
              <button className={sparkClassNames.iconButton} onClick={() => setSearchValue('')}>
                <CrossIcon className={sparkClassNames.icon}></CrossIcon>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
