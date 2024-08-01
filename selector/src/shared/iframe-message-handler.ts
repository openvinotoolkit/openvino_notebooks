import type { IAnalyticsMessage, IResizeMessage, IScrollMessage } from './iframe-message-emitter';

const notebooksSelectorElement = document.getElementById('notebooks-selector') as HTMLIFrameElement;

if (!notebooksSelectorElement) {
  throw new Error('Unable to find notebooks selector iframe element.');
}

function setIframeHeight(iframeElement: HTMLIFrameElement, heightPx: number): void {
  iframeElement.style.height = `${heightPx}px`;
}

function setInitialIframeHeight(iframeElement: HTMLIFrameElement): void {
  const iframeBodyHeight = iframeElement.contentDocument?.body?.offsetHeight;
  if (iframeBodyHeight) {
    setIframeHeight(iframeElement, iframeBodyHeight);
  }
}

window.onmessage = (message: MessageEvent<IResizeMessage | IScrollMessage | IAnalyticsMessage>) => {
  const { origin: allowedOrigin } = new URL(
    import.meta.env.PROD ? (import.meta.env.VITE_APP_LOCATION as string) : import.meta.url
  );

  if (message.origin !== allowedOrigin) {
    return;
  }

  if (message.data.type === 'resize' && message.data.height) {
    notebooksSelectorElement.style.height = message.data.height + 'px';
    return;
  }

  if (message.data.type === 'scroll') {
    notebooksSelectorElement.scrollIntoView({ behavior: 'smooth' });
    return;
  }

  if (message.data.type === 'analytics') {
    if (typeof window.wap_tms?.custom?.trackComponentClick === 'function') {
      window.wap_tms.custom.trackComponentClick(...message.data.args);
    } else {
      console.log('Analytics is not found on the host page.');
    }
    return;
  }
};

setInitialIframeHeight(notebooksSelectorElement);

export {};
