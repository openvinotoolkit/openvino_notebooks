import { type AdobeTrackFn } from './analytics/analytics';
import { isEmbedded } from './iframe-detector';
import { resolveTargetOrigin } from './target-origin';

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

const postToParent = (message: IResizeMessage | IScrollMessage | IAnalyticsMessage): void => {
  const targetOrigin = isEmbedded ? resolveTargetOrigin(document.referrer) : window.location.origin;
  if (!targetOrigin) {
    return; // Embedded by an untrusted page: don't leak data.
  }
  window.parent.postMessage(message, targetOrigin);
};

export const sendAnalyticsMessage = (...args: IAnalyticsMessage['args']): void => {
  postToParent({ type: 'analytics', args });
};

export const sendScrollMessage = (): void => {
  postToParent({ type: 'scroll' });
};

const report = () => {
  postToParent({ type: 'resize', height: document.body.offsetHeight });
};

new ResizeObserver(report).observe(document.body);

if (isEmbedded) {
  document.body.classList.add('embedded');
}
