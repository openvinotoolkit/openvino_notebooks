import './App.scss';

import { ContentSection } from '@components/ContentSection/ContentSection';
import { FiltersPanel } from '@components/FiltersPanel/FiltersPanel';
import { Header } from '@components/Header/Header';

function App(): JSX.Element {
  return (
    <>
      <Header />
      <main className="flex-col flex-1">
        <div className="flex flex-1">
          <FiltersPanel />
          <ContentSection />
        </div>
      </main>
    </>
  );
}

export default App;
