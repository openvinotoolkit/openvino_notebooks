import { useEffect } from 'react';

import { isEmbedded } from './iframe-detector';
import { CATEGORIES, LIBRARIES_VALUES, TASKS_VALUES } from './notebook-tags';
import { notebooksService } from './notebooks.service';
import { defaultSelectedTags, INotebooksSelector } from './notebooks-context';

type UrlPersistState = Pick<INotebooksSelector, 'searchValue' | 'selectedTags'>;

type Entries<T> = {
  [K in keyof T]: [K, T[K]];
}[keyof T][];

const OTHER_TAGS = await notebooksService.getOtherTags();

export function initializeSelectorUrlPersist(): UrlPersistState | null {
  const preservedUrlState = getUrlState();
  if (isEmbedded || !preservedUrlState) {
    return null;
  }
  return preservedUrlState;
}

export function useSelectorUrlPersist(notebooksSelector: INotebooksSelector): void {
  useEffect(() => {
    if (isEmbedded) {
      return;
    }
    const parent = window.parent;
    const stateSearchParams = toSearchParams(notebooksSelector).toString();
    const url = new URL(parent.location.toString());

    // no search params present - replace current state
    if (!url.search) {
      url.search = stateSearchParams;
      parent.history.replaceState(null, '', url);
      return;
    }

    // invoked on hitory pop state as well. Selector state was set according to search params - exit
    if (url.search.slice(1) === stateSearchParams) {
      return;
    }

    // new selector state - push new url search params
    url.search = stateSearchParams;
    parent.history.pushState(null, '', url);
  });

  if (isEmbedded) {
    return;
  }

  // listen for parent history state change
  // on each change set selector state according to url search params
  parent.onpopstate = () => {
    const state = getUrlState();
    console.log(state);

    if (state) {
      notebooksSelector.setSearchValue(state.searchValue);
      notebooksSelector.setSelectedTags(state.selectedTags);
    }
  };
}

export function getUrlState(): UrlPersistState | null {
  const parent = window.parent;
  if (isEmbedded || !parent.location.search) {
    return null;
  }
  const searchParams = new URLSearchParams(parent.location.search);
  return fromSearchParams(searchParams);
}

export function getUrlParamsWithSearch(searchValue: string): URLSearchParams {
  const urlSearchParams = new URLSearchParams();
  if (searchValue) {
    urlSearchParams.set('search', searchValue);
  }
  return urlSearchParams;
}

function toSearchParams(state: UrlPersistState): URLSearchParams {
  const urlSearchParams = getUrlParamsWithSearch(state.searchValue);

  for (const [key, filterValues] of Object.entries(state.selectedTags) as Entries<typeof state.selectedTags>) {
    if (filterValues.length) {
      const filterValuesString = filterValues.map((v) => v).join(',');
      urlSearchParams.set(key, filterValuesString);
    }
  }
  return urlSearchParams;
}

function fromSearchParams(urlSearchParams: URLSearchParams): UrlPersistState | null {
  function extractTagsValues<T extends string[]>(
    key: keyof UrlPersistState['selectedTags'],
    availableTagsValues?: T
  ): T {
    const filterTagValuesString = urlSearchParams.get(key);
    if (!filterTagValuesString) {
      return defaultSelectedTags[key] as T;
    }
    const filterTagValues = filterTagValuesString.split(',');
    return filterTagValues.filter((v) => {
      if (!availableTagsValues) {
        return true;
      }
      if (!availableTagsValues.includes(v)) {
        console.warn(`Invalid tag "${v}" for filter "${key}".`);
      }
      return availableTagsValues.includes(v);
    }) as T;
  }

  try {
    return {
      searchValue: urlSearchParams.get('search') || '',
      selectedTags: {
        categories: extractTagsValues('categories', Object.values(CATEGORIES)),
        tasks: extractTagsValues('tasks', TASKS_VALUES),
        libraries: extractTagsValues('libraries', LIBRARIES_VALUES),
        other: extractTagsValues('other', OTHER_TAGS),
      },
    };
  } catch (e) {
    console.warn(`Cannot restore state from url due to error:`, e);
    return null;
  }
}
