type ValidatedOS = 'ubuntu-20.04' | 'ubuntu-22.04' | 'windows-2019' | 'macos-12';

type ValidatedDevice = 'cpu' | 'gpu';

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
  status: {
    [OS in ValidatedOS]: {
      [Device in ValidatedDevice]: {
        [PythonVersion in ValidatedPythonVersion]: ValidationStatus | null;
      };
    };
  };
}
