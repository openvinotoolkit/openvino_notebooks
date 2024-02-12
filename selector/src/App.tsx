import { ContentSection } from '@components/ContentSection/ContentSection';
import { FiltersPanel } from '@components/FiltersPanel/FiltersPanel';
import { Header } from '@components/Header/Header';

import { NotebooksContext, useNotebooksSelector } from './shared/notebooks-context';

function App(): JSX.Element {
  const notebooksSelector = useNotebooksSelector();

  const isEmbedded = window !== window.parent;

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
