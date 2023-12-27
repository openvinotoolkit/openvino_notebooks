// @ts-check

import { CATEGORIES } from '../models/notebook-tags.js';

/**
 * @typedef {import('../models/notebook.js').INotebookMetadata} INotebookMetadata
 * @typedef {(v) => boolean} isValidFn
 * @typedef {(v) => string | null} ValidatorFn
 */

const toErrorMessage = ({ key, type, value }) => `'${key}' should be ${type}. Invalid value: ${JSON.stringify(value)}.`;

/** @type {(isValid: isValidFn, assertion: { key: string, type: string }) => ValidatorFn} */

const validate =
  (isValid, { key, type }) =>
  (v) =>
    isValid(v) ? null : toErrorMessage({ key, type, value: v }); // eslint-disable-line @typescript-eslint/no-unsafe-assignment

const isString = (v) => typeof v === 'string' || v instanceof String;

const isNotEmptyString = (v) => !!v && isString(v);

// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
const isUrl = (v) => URL.canParse(v);

// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
const isDate = (v) => isString(v) && !isNaN(new Date(v).getTime());

const isStringArray = (v) => Array.isArray(v) && v.every(isString);

/** @type {(f: isValidFn) => isValidFn} */
const Nullable = (f) => (v) => v === null || f(v);

/**
 * @param {INotebookMetadata['links']} links
 * @returns {ReturnType<ValidatorFn>}
 */
const linksValidator = ({ github, colab, binder }) => {
  const errors = [];
  if (!isUrl(github)) {
    errors.push(toErrorMessage({ key: 'links.github', type: 'a valid URL', value: github }));
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
  // TODO Add tags validator
  const errors = [];

  // TODO Consider validating that all keys exist

  for (const [key, value] of Object.entries(tags)) {
    if (!isStringArray(value)) {
      errors.push(toErrorMessage({ key: `tags.${key}`, type: 'a string array or empty array', value }));
    }
  }

  if (errors.length) {
    return errors.join('\n');
  }

  const { categories } = tags;

  const categoriesError = validateCategoriesTags(categories);
  if (categoriesError) {
    errors.push(categoriesError);
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
  if (!invalidTags.length) {
    return null;
  }
  return toErrorMessage({
    key: 'tags.categories',
    type: `a subset of ${JSON.stringify(validTags)}`,
    value: invalidTags,
  });
};

/** @type {Record<keyof INotebookMetadata, ValidatorFn>} */
const NOTEBOOK_METADATA_VALIDATORS = {
  title: validate(isNotEmptyString, { key: 'title', type: 'not empty string' }),
  description: validate(isString, { key: 'description', type: 'a string' }),
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
