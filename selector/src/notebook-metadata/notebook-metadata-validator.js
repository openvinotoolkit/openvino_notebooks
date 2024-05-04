// @ts-check

import { CATEGORIES, TASKS_VALUES } from '../shared/notebook-tags.js';

/**
 * @typedef {import('../shared/notebook-metadata.ts').INotebookMetadata} INotebookMetadata
 * @typedef {(v: any) => boolean} isValidFn
 * @typedef {(v: any) => string | null} ValidatorFn
 */

/** @type {(_: { key: string, type: string, value: any }) => string} */
const toErrorMessage = ({ key, type, value }) => `'${key}' should be ${type}. Invalid value: ${JSON.stringify(value)}.`;

/** @type {(isValid: isValidFn, assertion: { key: string, type: string }) => ValidatorFn} */
const validate =
  (isValid, { key, type }) =>
  (v) =>
    isValid(v) ? null : toErrorMessage({ key, type, value: v }); // eslint-disable-line @typescript-eslint/no-unsafe-assignment

const isString = (/** @type {any} */ v) => typeof v === 'string' || v instanceof String;

const isNotEmptyString = (/** @type {any} */ v) => !!v && isString(v);

const isUrl = (/** @type {string} */ v) => URL.canParse(v);

const isDate = (/** @type {string} */ v) => isString(v) && !isNaN(new Date(v).getTime());

const isStringArray = (/** @type {any[]} */ v) => Array.isArray(v) && v.every(isString);

/** @type {(f: isValidFn) => isValidFn} */
const Nullable = (f) => (v) => v === null || f(v);

/**
 * @param {INotebookMetadata['links']} links
 * @returns {ReturnType<ValidatorFn>}
 */
const linksValidator = ({ github, docs, colab, binder }) => {
  const errors = [];
  if (!isUrl(github)) {
    errors.push(toErrorMessage({ key: 'links.github', type: 'a valid URL', value: github }));
  }
  if (!Nullable(isUrl)(docs)) {
    errors.push(toErrorMessage({ key: 'links.docs', type: 'a valid URL or null', value: docs }));
  }
  if (!Nullable(isUrl)(colab)) {
    errors.push(toErrorMessage({ key: 'links.colab', type: 'a valid URL or null', value: colab }));
  }
  if (!Nullable(isUrl)(binder)) {
    errors.push(toErrorMessage({ key: 'links.binder', type: 'a valid URL or null', value: binder }));
  }
  return errors.length ? errors.join('\n') : null;
};

/**
 * @param {INotebookMetadata['tags']} tags
 * @returns {ReturnType<ValidatorFn>}
 */
const tagsValidator = (tags) => {
  const errors = [];

  /** @type {(keyof typeof tags)[]} */
  const tagsKeys = ['categories', 'tasks', 'libraries', 'other'];

  for (const key of tagsKeys) {
    const value = tags[key];
    if (!isStringArray(value)) {
      errors.push(toErrorMessage({ key: `tags.${key}`, type: 'a string array or empty array', value }));
    }
  }

  if (errors.length) {
    return errors.join('\n');
  }

  const { categories, tasks } = tags;

  const categoriesError = validateCategoriesTags(categories);
  if (categoriesError) {
    errors.push(categoriesError);
  }

  const tasksError = validateTasksTags(tasks);
  if (tasksError) {
    errors.push(tasksError);
  }

  return errors.length ? errors.join('\n') : null;
};

/**
 * @param {INotebookMetadata['tags']['categories']} categories
 * @returns {ReturnType<ValidatorFn>}
 */
const validateCategoriesTags = (categories) => {
  const validTags = Object.values(CATEGORIES);
  const invalidTags = categories.filter((tag) => !validTags.includes(tag));
  if (categories.length && !invalidTags.length) {
    return null;
  }
  return toErrorMessage({
    key: 'tags.categories',
    type: `a subset of ${JSON.stringify(validTags)}`,
    value: invalidTags,
  });
};

/**
 * @param {INotebookMetadata['tags']['tasks']} tasks
 * @returns {ReturnType<ValidatorFn>}
 */
const validateTasksTags = (tasks) => {
  const validTags = TASKS_VALUES;
  const invalidTags = tasks.filter((tag) => !validTags.includes(tag));
  if (tasks.length && !invalidTags.length) {
    return null;
  }
  return toErrorMessage({
    key: 'tags.tasks',
    type: `a subset of ${JSON.stringify(validTags)}`,
    value: invalidTags,
  });
};

/** @type {Record<keyof INotebookMetadata, ValidatorFn>} */
const NOTEBOOK_METADATA_VALIDATORS = {
  title: validate(isNotEmptyString, { key: 'title', type: 'not empty string' }),
  path: validate(isNotEmptyString, { key: 'path', type: 'not empty string' }),
  imageUrl: validate(Nullable(isUrl), { key: 'imageUrl', type: 'a valid URL or null' }),
  createdDate: validate(isDate, { key: 'createdDate', type: 'a valid Date string' }),
  modifiedDate: validate(isDate, { key: 'modifiedDate', type: 'a valid Date string' }),
  links: linksValidator,
  tags: tagsValidator,
};

export class NotebookMetadataValidationError extends Error {}

/**
 * Validates notebook metadata object
 *
 * @param {INotebookMetadata} metadata
 * @throws {NotebookMetadataValidationError} Error message containing all metadata invalid properties
 * @returns {void}
 */
export function validateNotebookMetadata(metadata) {
  const errors = [];
  const entries = /** @type {[keyof INotebookMetadata, any][]} */ (Object.entries(metadata));
  for (const [key, value] of entries) {
    const validator = NOTEBOOK_METADATA_VALIDATORS[key];
    if (!validator) {
      errors.push(`Unknown metadata property "${key}".`);
      continue;
    }
    const error = validator(value);
    if (error) {
      errors.push(error);
    }
  }
  if (errors.length) {
    throw new NotebookMetadataValidationError(
      `The following notebook metadata properties are not valid:\n${errors.join('\n')}\n`
    );
  }
}
