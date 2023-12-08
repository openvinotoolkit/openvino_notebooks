import './App.scss';

import { ContentSection } from '@components/ContentSection/ContentSection';
import { FiltersSection } from '@components/FiltersSection/FiltersSection';
import { Header } from '@components/Header/Header';

function App(): JSX.Element {
  return (
    <>
      <Header />
      <main className="flex-col flex-1">
        <div className="flex">
          <FiltersSection />
          <ContentSection />
        </div>
      </main>
    </>
  );
}

export default App;
