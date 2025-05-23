/**
 * Intel Global Header Footer (50recode.2)
 */

export function addIGHF(): void {
  addPrefetch();
  addLoader();
}

function addPrefetch(): void {
  const meta = document.createElement('meta');
  meta.httpEquiv = 'x-dns-prefetch-control';
  meta.content = 'on';

  const links: HTMLLinkElement[] = [
    createLinkElement({ rel: 'dns-prefetch', href: 'https://www.intel.com', pr: '1.0' }),
    createLinkElement({ rel: 'dns-prefetch', href: 'https://www.google-analytics.com', pr: '1.0' }),
    createLinkElement({ rel: 'preconnect', href: 'https://www.intel.com', pr: '1.0', crossOrigin: 'anonymous' }),
    createLinkElement({
      rel: 'preconnect',
      href: 'https://www.google-analytics.com',
      pr: '1.0',
      crossOrigin: 'anonymous',
    }),
  ];

  const commentStart = document.createComment('<!--IGHF (Performance tweaks for Mobile and Slow connections)-->');
  const commentEnd = document.createComment('<!--/IGHF-->');

  for (const element of [commentStart, meta, ...links, commentEnd]) {
    document.head.appendChild(element);
  }
}

function createLinkElement({ rel, href, pr, crossOrigin }: Partial<HTMLLinkElement> & { pr: string }): HTMLLinkElement {
  const link = document.createElement('link');
  if (rel) {
    link.rel = rel;
  }
  if (href) {
    link.href = href;
  }
  link.setAttribute('pr', pr);
  if (crossOrigin) {
    link.crossOrigin = crossOrigin;
  }
  return link;
}

function createScriptElement({ src, async, defer }: Partial<HTMLScriptElement>): HTMLScriptElement {
  const script = document.createElement('script');
  script.type = 'text/javascript';
  if (src) {
    script.src = src;
  }
  if (async) {
    script.async = async;
  }
  if (defer) {
    script.defer = defer;
  }
  return script;
}

function addLoader(): void {
  const commentStart = document.createComment('<!--IGHF Loader-->');
  const commentEnd = document.createComment('<!--/IGHF Loader-->');
  const headerChooserScript = createScriptElement({
    src: 'https://www.intel.com/ighf/50recode.2/js/headerchooser.js',
    async: true,
  });

  for (const element of [commentStart, headerChooserScript, commentEnd]) {
    document.head.appendChild(element);
  }
}

/**
 * IGHF Header is configured in Header component
 */
export function setIntelnavRenderSettings(): void {
  if (!window.INTELNAV) {
    window.INTELNAV = {};
  }
  window.INTELNAV.renderSettings = {
    version: '2.0 - 03/12/2017 08:00:00',
    textDirection: 'ltr',
    culture: 'en_US',
    OutputId: 'default',
  };
}

/**
 * IGHF Footer is configured in Footer component
 */
export function setIntelnavRenderSettingsFooter(): void {
  if (!window.INTELNAV) {
    window.INTELNAV = {};
  }
  window.INTELNAV.renderSettingsFooter = {
    version: '2.0 - 03/12/2017 08:00:00',
    OutputId: 'gf_default',
  };
}
