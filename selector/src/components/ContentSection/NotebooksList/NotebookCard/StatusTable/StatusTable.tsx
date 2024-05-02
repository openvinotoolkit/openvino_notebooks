import './StatusTable.scss';

import CheckIcon from '@/assets/images/check.svg?react';
import CrossIcon from '@/assets/images/cross.svg?react';
import DenyIcon from '@/assets/images/deny.svg?react';
import PythonIcon from '@/assets/images/python.svg?react';
import TimeoutIcon from '@/assets/images/timeout.svg?react';
import { Tooltip } from '@/components/shared/Tooltip/Tooltip';

type CheckPythonStatus = 'SUCCESS' | 'FAILED' | 'TIMEOUT' | 'NOT_RUN' | 'SKIPPED' | 'EMPTY' | null;

const getStatusIcon = (status: CheckPythonStatus): JSX.Element => {
  if (status === 'SUCCESS') {
    return (
      <Tooltip content="Passed">
        <CheckIcon className="status-icon status-icon-success" />
      </Tooltip>
    );
  }
  if (status === 'FAILED') {
    return (
      <Tooltip content="Failed">
        <CrossIcon className="status-icon status-icon-failed" />
      </Tooltip>
    );
  }
  if (status === 'TIMEOUT') {
    return (
      <Tooltip content="Exceeded time limits (2 hours)">
        <TimeoutIcon className="status-icon status-icon-warn" />
      </Tooltip>
    );
  }
  return (
    <Tooltip content="Skipped">
      <DenyIcon className="status-icon status-icon-skipped" />
    </Tooltip>
  );
};

export const StatusTable = () => (
  <div className="status-table spark-font-75">
    <div className="device-header">
      <div className="cell">CPU</div>
    </div>
    <div className="python-versions">
      <div className="cell">
        <PythonIcon className="python-icon" />
        3.8
      </div>
      <div className="cell">
        <PythonIcon className="python-icon" />
        3.9
      </div>
      <div className="cell">
        <PythonIcon className="python-icon" />
        3.10
      </div>
    </div>
    <div className="os-header">
      <div className="cell">OS</div>
    </div>
    <div className="os-names">
      <div className="cell">ubuntu-20.04</div>
      <div className="cell">ubuntu-22.04</div>
      <div className="cell">windows-2019</div>
      <div className="cell">macos-12</div>
    </div>
    <div className="statuses">
      {/* Row OS 1 */}
      <div className="cell">{getStatusIcon('SUCCESS')}</div>
      <div className="cell">{getStatusIcon('TIMEOUT')}</div>
      <div className="cell">{getStatusIcon('SKIPPED')}</div>
      {/* Row OS 2 */}
      <div className="cell">{getStatusIcon('SUCCESS')}</div>
      <div className="cell">{getStatusIcon('SUCCESS')}</div>
      <div className="cell">{getStatusIcon('SUCCESS')}</div>
      {/* Row OS 3 */}
      <div className="cell">{getStatusIcon('FAILED')}</div>
      <div className="cell">{getStatusIcon('SUCCESS')}</div>
      <div className="cell">{getStatusIcon('SKIPPED')}</div>
      {/* Row OS 4 */}
      <div className="cell">{getStatusIcon('SUCCESS')}</div>
      <div className="cell">{getStatusIcon('SKIPPED')}</div>
      <div className="cell">{getStatusIcon('FAILED')}</div>
    </div>
  </div>
);
