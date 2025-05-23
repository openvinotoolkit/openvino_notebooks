declare global {
  interface Window {
    INTELNAV?: {
      renderSettings?: {
        version: string;
        textDirection: string;
        culture: string;
        OutputId: string;
      };
      renderSettingsFooter?: {
        version: string;
        OutputId: string;
      };
    };
  }
}

export {};
