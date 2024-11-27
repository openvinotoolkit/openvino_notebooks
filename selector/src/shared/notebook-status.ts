type ValidatedOS = 'ubuntu-20.04' | 'ubuntu-22.04' | 'windows-2019' | 'macos-13';

type ValidatedDevice = 'cpu' | 'gpu';

export const VALIDATED_PYTHON_VERSIONS = ['3.9', '3.10', '3.11', '3.12'] as const;

type ValidatedPythonVersion = (typeof VALIDATED_PYTHON_VERSIONS)[number];

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
