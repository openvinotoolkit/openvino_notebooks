import './Footer.scss';

import { setIntelnavRenderSettingsFooter } from '@/shared/ighf/ighf';

export const Footer = (): JSX.Element => {
  setIntelnavRenderSettingsFooter();
  return (
    <footer>
      {/* <!--IGHF Footer-->  */}
      {/* <!--GAATversion='50recode.2' date='09/11/2017 08:00:00' Version='2.0':CharacterEncoding:utf8-->  */}
      <div id="recode50footer"></div>
      <noscript>
        <div id="smallfootprint-footer">
          <ul>
            <li>©Intel Corporation</li>
            <li>
              <a href="https://www.intel.com/content/www/us/en/legal/terms-of-use.html" target="">
                Terms of Use
              </a>
            </li>
            <li>
              <a href="https://www.intel.com/content/www/us/en/legal/trademarks.html" target="">
                *Trademarks
              </a>
            </li>
            <li>
              <a href="https://www.intel.com/content/www/us/en/privacy/intel-privacy-notice.html" target="">
                Privacy
              </a>
            </li>
            <li>
              <a href="https://www.intel.com/content/www/us/en/privacy/intel-cookie-notice.html" target="">
                Cookies
              </a>
            </li>
            <li>
              <a
                href="https://www.intel.com/content/www/us/en/policy/policy-human-trafficking-and-slavery.html"
                target=""
              >
                Supply Chain Transparency
              </a>
            </li>
            <li>
              <a href="https://www.intel.com/content/www/us/en/siteindex.html" target="">
                Site Map
              </a>
            </li>
          </ul>
        </div>
      </noscript>
      {/* <!--/IGHF Footer--> */}
    </footer>
  );
};
