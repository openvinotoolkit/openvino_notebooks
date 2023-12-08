import './Header.scss';

import logo from '@assets/images/logo.svg';

export const Header = (): JSX.Element => {
  return (
    <header role="banner" className="spark-header spark-header-size-s">
      <div className="spark-header-brand">
        <div className="spark-header-brand-logoimg">
          <img alt="OpenVINO" src={logo}></img>
        </div>
      </div>
      <nav className="spark-header-region-start">
        <div className="spark-header-project-name">Notebooks</div>
      </nav>
    </header>
  );
};
