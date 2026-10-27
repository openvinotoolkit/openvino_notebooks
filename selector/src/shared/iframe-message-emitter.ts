import { type AdobeTrackFn } from './analytics/analytics';
import { isEmbedded } from './iframe-detector';

// Optional: when VITE_PARENT_ORIGIN is set, the messages are only delivered to a parent page with that origin.
// When it isn't set they go to any parent like before, so existing embeds keep working.
const targetOrigin = (import.meta.env.VITE_PARENT_ORIGIN as string | undefined) || '*';

export interface IResizeMessage {
  type: 'resize';
  height: number;
}

export interface IScrollMessage {
  type: 'scroll';
}

export interface IAnalyticsMessage {
  type: 'analytics';
  args: Parameters<AdobeTrackFn>;
}

export const sendAnalyticsMessage = (...args: IAnalyticsMessage['args']): void => {
  const message: IAnalyticsMessage = {
    type: 'analytics',
    args,
  };
  window.parent.postMessage(message, targetOrigin);
};

export const sendScrollMessage = (): void => {
  const message: IScrollMessage = {
    type: 'scroll',
  };
  window.parent.postMessage(message, targetOrigin);
};

const report = () => {
  const message: IResizeMessage = {
    type: 'resize',
    height: document.body.offsetHeight,
  };
  window.parent.postMessage(message, targetOrigin);
};

new ResizeObserver(report).observe(document.body);

if (isEmbedded) {
  document.body.classList.add('embedded');
}
