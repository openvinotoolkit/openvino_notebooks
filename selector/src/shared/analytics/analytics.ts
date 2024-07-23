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
}

enum COMPONENT {
  NAVIGATE = 'ov-notebooks:navigate',
  COPY_LINK = 'ov-notebooks:copy-link',
  // SHOW_STATUS = 'ov-notebooks:show-status',
  // FILTER = 'ov-notebooks:filter',
}

interface AdobeTrackFn {
  (componentName: COMPONENT, value: string): void;
}

function getAdobeAnalyticsFunction(window: Window): AdobeTrackFn | null {
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

  private _send = (component: COMPONENT, value: string) => {
    if (!this._window) {
      this._consoleNotification.notInitialized();
      return;
    }

    if (import.meta.env.DEV) {
      this._consoleNotification.devMode();
      console.log(`[Analytics] Component: ${component}. Value: ${value}.`);
    }

    const adobeSend = getAdobeAnalyticsFunction(this._window);

    if (!adobeSend) {
      this._consoleNotification.notFound();
      return;
    }

    try {
      adobeSend(component, value);
    } catch (e) {
      console.error(e);
    }
  };

  sendNavigateEvent(destination: string): void {
    this._send(COMPONENT.NAVIGATE, `${destination}`);
  }

  sendCopyLinkEvent(notebookPath: string): void {
    this._send(COMPONENT.COPY_LINK, notebookPath);
  }
}

export const analytics = new Analytics();
