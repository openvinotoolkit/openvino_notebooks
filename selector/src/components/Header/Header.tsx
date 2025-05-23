import './Header.scss';

import logo from '@assets/images/logo.svg';

import { setIntelnavRenderSettings } from '@/shared/ighf/ighf';

export const Header = (): JSX.Element => {
  setIntelnavRenderSettings();
  return (
    <>
      {/* <!--IGHF Header--> */}
      {/* <!--GAATversion='50recode.2' date='09/11/2017 08:00:00' Version='2.0':CharacterEncoding:utf8--> */}
      <div id="recode50header" className="no-animate" style={{ display: 'none' }}></div>
      {/* <!--/IGHF Header--> */}

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
    </>
  );
};
