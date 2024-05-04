const filtersOpenedClassName = 'filters-opened';

export const openFiltersPanel = (): void => {
  document.body.classList.add(filtersOpenedClassName);
};

export const closeFiltersPanel = (): void => {
  document.body.classList.remove(filtersOpenedClassName);
};
