import './StatusTable.scss';

import CheckIcon from '@/assets/images/check.svg?react';
import CrossIcon from '@/assets/images/cross.svg?react';
import DenyIcon from '@/assets/images/deny.svg?react';
import PythonIcon from '@/assets/images/python.svg?react';
import TimeoutIcon from '@/assets/images/timeout.svg?react';
import { Tooltip } from '@/components/shared/Tooltip/Tooltip';
import { VALIDATED_PYTHON_VERSIONS, ValidationStatus } from '@/shared/notebook-status';
import { NotebookItem } from '@/shared/notebooks.service';

const getStatusIcon = (status: ValidationStatus | null): JSX.Element => {
  if (status === ValidationStatus.SUCCESS) {
    return (
      <Tooltip content="Passed">
        <CheckIcon className="status-icon status-icon-success" />
      </Tooltip>
    );
  }
  if (status === ValidationStatus.FAILED) {
    return (
      <Tooltip content="Failed">
        <CrossIcon className="status-icon status-icon-failed" />
      </Tooltip>
    );
  }
  if (status === ValidationStatus.TIMEOUT) {
    return (
      <Tooltip content="Exceeded time limits (2 hours)">
        <TimeoutIcon className="status-icon status-icon-warn" />
      </Tooltip>
    );
  }
  if ([ValidationStatus.SKIPPED, ValidationStatus.NOT_RUN, ValidationStatus.EMPTY].includes(status!)) {
    return (
      <Tooltip content="Skipped">
        <DenyIcon className="status-icon status-icon-skipped" />
      </Tooltip>
    );
  }
  return <Tooltip content="Not available">N/A</Tooltip>;
};

const pythonVersionsCells = (
  <>
    {VALIDATED_PYTHON_VERSIONS.map((v) => (
      <div key={`python-${v}`} className="cell">
        <PythonIcon className="python-icon" />
        {v}
      </div>
    ))}
  </>
);

type StatusTableProps = {
  status: NonNullable<NotebookItem['status']>;
};

export const StatusTable = ({ status }: StatusTableProps) => {
  const osOptions = Object.keys(status) as (keyof typeof status)[];
  const cpuStatuses = osOptions.map((os) => Object.values(status[os]['cpu'])).flat();
  const gpuStatuses = osOptions.map((os) => Object.values(status[os]['gpu'])).flat();
  const showGPU = false;

  return (
    <div className="status-table spark-font-75">
      <div className="cpu-device-header device-header">
        <div className="cell">CPU</div>
      </div>
      {showGPU && (
        <div className="gpu-device-header device-header">
          <div className="cell">GPU</div>
        </div>
      )}
      <div className="cpu-python-versions python-versions">{pythonVersionsCells}</div>
      {showGPU && <div className="gpu-python-versions python-versions">{pythonVersionsCells}</div>}
      <div className="os-header">
        <div className="cell">OS</div>
      </div>
      <div className="os-names">
        {osOptions.map((os) => (
          <div key={os} className="cell">
            {os}
          </div>
        ))}
      </div>
      <div className="cpu-statuses statuses">
        {cpuStatuses.map((v, i) => (
          <div key={`status-cpu-${i}`} className="cell">
            {getStatusIcon(v)}
          </div>
        ))}
      </div>
      {showGPU && (
        <div className="gpu-statuses statuses">
          {gpuStatuses.map((v, i) => (
            <div key={`status-gpu-${i}`} className="cell">
              {getStatusIcon(v)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
