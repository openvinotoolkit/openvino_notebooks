import './Pagination.scss';

import ChevronIcon from '@assets/images/chevron.svg?react';
import ChevronDoubleIcon from '@assets/images/chevron-double.svg?react';
import { FunctionComponent, HTMLProps } from 'react';

import { Button } from '../Button/Button';
import { Dropdown } from '../Dropdown/Dropdown';

const rotateIconHOC = (
  WrappedComponent: FunctionComponent<HTMLProps<SVGSVGElement>>,
  value: number
): FunctionComponent => {
  return function RotatedComponent(props) {
    return <WrappedComponent {...props} style={{ transform: `rotate(${value}deg)` }}></WrappedComponent>;
  };
};

type PaginationProps = {
  itemsPerPageOptions: number[];
  itemsPerPage: number;
  page: number;
  totalPages: number;
  onChangePage: (value: number) => void;
  onChangeItemsPerPage: (value: number) => void;
  className?: string;
};

export const Pagination = ({
  itemsPerPageOptions,
  itemsPerPage,
  onChangePage,
  onChangeItemsPerPage,
  page,
  totalPages,
  className = '',
}: PaginationProps): JSX.Element => {
  return (
    <div className={`pagination ${className}`}>
      <div className="pagination-control">
        <span className="spark-font-75">Items per page</span>
        <Dropdown
          className="items-per-page-dropdown"
          direction="top"
          options={itemsPerPageOptions.map((v) => v.toString())}
          selectedOption={itemsPerPage.toString()}
          onSelect={(v) => {
            onChangeItemsPerPage(Number(v));
            onChangePage(1);
          }}
        />
      </div>
      <div className="pagination-list">
        <Button
          variant="secondary"
          icon={rotateIconHOC(ChevronDoubleIcon, 90)}
          disabled={page === 1}
          onClick={() => onChangePage(1)}
        ></Button>
        <Button
          variant="secondary"
          icon={rotateIconHOC(ChevronIcon, 90)}
          disabled={page === 1}
          onClick={() => onChangePage(page - 1)}
        ></Button>
        <span className="spark-font-75">
          Page {page} of {totalPages}
        </span>
        <Button
          variant="secondary"
          icon={rotateIconHOC(ChevronIcon, -90)}
          disabled={page === totalPages}
          onClick={() => onChangePage(page + 1)}
        ></Button>
        <Button
          variant="secondary"
          icon={rotateIconHOC(ChevronDoubleIcon, -90)}
          onClick={() => onChangePage(totalPages)}
          disabled={page === totalPages}
        ></Button>
      </div>
    </div>
  );
};
