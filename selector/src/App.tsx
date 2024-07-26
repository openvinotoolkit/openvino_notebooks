import { ContentSection } from '@components/ContentSection/ContentSection';
import { FiltersPanel } from '@components/FiltersPanel/FiltersPanel';
import { Footer } from '@components/Footer/Footer';
import { Header } from '@components/Header/Header';

import { analytics } from '@/shared/analytics/analytics';
import { isEmbedded } from '@/shared/iframe-detector';
import { NotebooksContext, useNotebooksSelector } from '@/shared/notebooks-context';
import { initializeSelectorUrlPersist, useSelectorUrlPersist } from '@/shared/selectorUrlPersist';

const initialState = initializeSelectorUrlPersist();
analytics.initialize(window.parent);

function App(): JSX.Element {
  const notebooksSelector = useNotebooksSelector(initialState);
  useSelectorUrlPersist(notebooksSelector);

  return (
    <>
      {!isEmbedded && <Header />}
      <main className="flex-col flex-1">
        <div className="flex flex-1">
          <NotebooksContext.Provider value={notebooksSelector}>
            <FiltersPanel />
            <ContentSection />
          </NotebooksContext.Provider>
        </div>
      </main>
      {!isEmbedded && <Footer />}
    </>
  );
}

export default App;
