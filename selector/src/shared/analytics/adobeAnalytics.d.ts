declare global {
  interface Window {
    wap_tms?: {
      custom?: {
        trackComponentClick?: (componentName: string, value: string) => void;
      };
    };
    wapLocalCode?: string;
    wapSection?: string;
  }
}

export {};
