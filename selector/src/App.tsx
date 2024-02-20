import { ContentSection } from '@components/ContentSection/ContentSection';
import { FiltersPanel } from '@components/FiltersPanel/FiltersPanel';
import { Header } from '@components/Header/Header';

import { isEmbedded } from '@/shared/iframe-detector';

import { NotebooksContext, useNotebooksSelector } from './shared/notebooks-context';

function App(): JSX.Element {
  const notebooksSelector = useNotebooksSelector();

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
    </>
  );
}

export default App;
