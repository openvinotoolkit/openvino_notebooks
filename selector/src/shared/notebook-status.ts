type ValidatedOS = 'ubuntu-20.04' | 'ubuntu-22.04' | 'windows-2019' | 'macos-12';

type ValidatedPythonVersion = '3.8' | '3.9' | '3.10';

export enum ValidationStatus {
  SUCCESS = 'SUCCESS',
  FAILED = 'FAILED',
  TIMEOUT = 'TIMEOUT',
  SKIPPED = 'SKIPPED',
  NOT_RUN = 'NOT_RUN',
  EMPTY = 'EMPTY',
}

export interface INotebookStatus {
  name: string;
  statuses: {
    [OS in ValidatedOS]: {
      [PythonVersion in ValidatedPythonVersion]: ValidationStatus | null;
    };
  };
}
