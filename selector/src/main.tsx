import './styles/styles.scss';
import '@/shared/iframe-message-emitter';

import React from 'react';
import ReactDOM from 'react-dom/client';

import { addAnalyticsScript } from '@/shared/analytics/analytics.ts';
import { addIGHF } from '@/shared/ighf/ighf.ts';

import App from './App.tsx';
import { isEmbedded } from './shared/iframe-detector.ts';

if (!isEmbedded) {
  addAnalyticsScript();
  addIGHF();
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
