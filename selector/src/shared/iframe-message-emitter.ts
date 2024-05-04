import { isEmbedded } from './iframe-detector';

export interface IResizeMessage {
  type: 'resize';
  height: number;
}

export interface IScrollMessage {
  type: 'scroll';
}

export const sendScrollMessage = (): void => {
  const message: IScrollMessage = {
    type: 'scroll',
  };
  window.parent.postMessage(message, '*');
};

const report = () => {
  const message: IResizeMessage = {
    type: 'resize',
    height: document.body.offsetHeight,
  };
  window.parent.postMessage(message, '*');
};

new ResizeObserver(report).observe(document.body);

if (isEmbedded) {
  document.body.classList.add('embedded');
}
