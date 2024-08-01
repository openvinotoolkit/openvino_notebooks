import './Footer.scss';

export const Footer = (): JSX.Element => {
  return (
    <footer>
      <div className="footer-container">
        <ul className="footer-list">
          <li>© Intel Corporation</li>
          <li>
            <a
              href="https://docs.openvino.ai/2024/about-openvino/additional-resources/terms-of-use.html"
              target="_blank"
              rel="noreferrer"
            >
              Terms of Use
            </a>
          </li>
          <li>
            <a
              href="https://www.intel.com/content/www/us/en/privacy/intel-cookie-notice.html"
              target="_blank"
              data-cookie-notice="true"
              rel="noreferrer"
            >
              Cookies
            </a>
          </li>
          <li>
            <a
              href="https://www.intel.com/content/www/us/en/privacy/intel-privacy-notice.html"
              target="_blank"
              rel="noreferrer"
            >
              Privacy
            </a>
          </li>
          <li data-wap_ref="dns" id="wap_dns">
            <a href="/#" target="_parent">
              Your Privacy Choices
            </a>
          </li>
          <li id="footer_it_problem_fix_do_not_remove"></li>
          <li data-wap_ref="nac" id="wap_nac">
            <a
              href="https://www.intel.com/content/www/us/en/privacy/privacy-residents-certain-states.html"
              target="_parent"
            >
              Notice at Collection
            </a>
          </li>
        </ul>
      </div>
    </footer>
  );
};
