import { isEmbedded } from '../iframe-detector';
import { sendAnalyticsMessage } from '../iframe-message-emitter';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const once = function <T extends (...args: any[]) => any>(fn: T) {
  let result: ReturnType<T>;
  let invoked = false;

  return function (...args: Parameters<T>): ReturnType<T> {
    if (invoked) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return result;
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    result = fn(args);
    invoked = true;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return result;
  };
};

export function addAnalyticsScript(): void {
  const host = window.document.location.protocol == 'http:' ? 'http://www.intel.com' : 'https://www.intel.com';
  const url = host + '/content/dam/www/global/wap/tms-loader.js'; // wap file url
  const scriptElement = document.createElement('script');
  scriptElement.type = 'text/javascript';
  scriptElement.async = true;
  scriptElement.src = url;
  const headElement = document.getElementsByTagName('head')[0];
  headElement.appendChild(scriptElement);
  // Set analytics vars
  window.wapLocalCode = 'us-en';
  window.wapSection = 'openvinotoolkit';
}

enum COMPONENT {
  NAVIGATE = 'ov-notebooks:navigate',
  COPY_LINK = 'ov-notebooks:copy-link',
  FILTER = 'ov-notebooks:filter',
  SEARCH = 'ov-notebooks:search',
}

export type AdobeTrackFn = (componentName: COMPONENT, label: string, detail?: string) => void;

function getAdobeAnalyticsFunction(window: Window): AdobeTrackFn | null {
  if (isEmbedded) {
    return sendAnalyticsMessage;
  }

  if (typeof window.wap_tms?.custom?.trackComponentClick !== 'function') {
    return null;
  }

  return window.wap_tms.custom.trackComponentClick.bind(window.wap_tms.custom);
}

class Analytics {
  private _window?: Window;
  private _consoleNotification = {
    notInitialized: once(() => console.log('Analytics is not initialized.')),
    notFound: once(() => console.log('Analytics is not found on the page.')),
    devMode: once(() => console.log('Analytics is in dev mode.')),
  };

  initialize(window: Window) {
    this._window = window;
  }

  private _send: AdobeTrackFn = (component, label, detail) => {
    if (!this._window) {
      this._consoleNotification.notInitialized();
      return;
    }

    if (import.meta.env.DEV) {
      this._consoleNotification.devMode();
      console.log(`[Analytics] Component: ${component}\n\tLabel: ${label}\n\tDetail: ${detail}`);
    }

    const adobeSend = getAdobeAnalyticsFunction(this._window);

    if (!adobeSend) {
      this._consoleNotification.notFound();
      return;
    }

    try {
      adobeSend(component, label, detail);
    } catch (e) {
      console.error(e);
    }
  };

  sendNavigateEvent(notebookPath: string, url: string): void {
    this._send(COMPONENT.NAVIGATE, notebookPath, url);
  }

  sendCopyLinkEvent(notebookPath: string): void {
    this._send(COMPONENT.COPY_LINK, notebookPath);
  }

  sendFilterEvent(filterOption: string) {
    this._send(COMPONENT.FILTER, filterOption);
  }

  sendSearchEvent(searchValue: string) {
    this._send(COMPONENT.SEARCH, searchValue);
  }
}

export const analytics = new Analytics();
