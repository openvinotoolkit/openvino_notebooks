# Nightly dashboard data

This package produces the data consumed by the **notebooks nightly dashboard** (a separate GitHub Pages project).
It is independent from the other nightly artifacts (`test_report-*`, `notebooks-status-map.json`), which are left unchanged.

## Pipeline

```
treon_nightly.yml
├─ collect_notebooks      split_notebooks.py --plan_file          ─► artifact nightly-plan            (plan.json)
├─ build_treon_* (matrix) validate_notebooks.py --dashboard_dir   ─► artifact nightly-fragment-<device>-<os>-<python>-<batch>
│                                                                     fragment.json, logs/<notebook>.log, envs/<env_id>.txt
└─ build_dashboard_data   nightly_report/build_report.py          ─► artifact nightly-dashboard-data  (retention 90 days)
                                                                      nightly-report.json, environments.json, summary.md,
                                                                      nightly-report.schema.json
```

* **Fragments** are written by every test job and rewritten after each notebook, so cancelled or crashed jobs still report what they
  completed. Logs are kept only for notebooks that did not pass (capped at 2 MB, head and tail preserved).
* **build_report.py** joins fragments with the batch plan, the GitHub jobs/artifacts API and the repository checkout. Notebooks of jobs
  that died before tests started are reported as `not_run` with the failed step, so infrastructure problems are visible.
  Matrix jobs skipped or cancelled before they started are expanded from the workflow file (`--workflow_file`).
  Data problems never fail the build; they are listed in `generator.warnings`. The step exits non-zero only if the produced report
  violates the schema.
* `summary.md` is also published as the job summary of the workflow run.
* `pull_request` runs of the workflow test only the notebooks of [`.ci/nightly_sample_notebooks.txt`](../nightly_sample_notebooks.txt)
  (`split_notebooks.py --notebooks_list`) and do not upload `notebooks-status-map.json`; their reports have `run.event == "pull_request"`.

The files are described by JSON Schemas in [`schema/`](./schema): `nightly-report.schema.json` (the dashboard contract),
`fragment.schema.json` and `plan.schema.json` (internal).

## Ingestion by the dashboard (pull model)

1. List the artifacts:
   `GET https://api.github.com/repos/openvinotoolkit/openvino_notebooks/actions/artifacts?name=nightly-dashboard-data&per_page=100`
2. Keep artifacts with `expired == false`. For the official trend keep `workflow_run.head_branch == "latest"` and, after download,
   `run.event == "schedule"` (manual `workflow_dispatch` runs are useful but should not be mixed into the trend).
3. Skip `workflow_run.id` values that were already ingested. If a run is re-attempted, keep the report with the highest `run.attempt`.
4. Download `archive_download_url` with any token (the `GITHUB_TOKEN` of the dashboard repository is sufficient for this public repository),
   unzip and read `nightly-report.json`.
5. Reject reports whose `schema_version` has an unknown **major** version; minor versions only add optional fields.
6. Poll at least daily: artifacts are kept for 90 days at most (the repository setting can shorten this).

Size: roughly 1 MB of JSON per night. Keep `results` (and `summary`) long-term; `failures` text and `environments.json` can be pruned
after some weeks.

## `nightly-report.json` overview

| Key | Content |
|---|---|
| `run` | Run id/attempt/number, event, branch, `nightly_date` (UTC date, x-axis), timings, conclusion, run URL, tested commit, inputs (OV branch, docker tag, timeout, batches, ignore lists). |
| `configs[]` | Test configurations `<device>-<os>-py<python>` with runner label, container image, job ids. |
| `jobs[]` | Test jobs: conclusion, timings, runner, links (job, notebook test step, logs artifact), failed step, whether tests started, fragment presence. Identified by (`id`, `config_id`, `batch`); results join on the same fields. |
| `notebooks[]` | Catalog at the tested commit: path, title, tags, `dir_tree_sha` (changes iff the notebook directory changes), last modifying commit, links (source at commit, latest, history). |
| `results[]` | One row per planned (notebook, config): status, skip/not-run reason, start time, duration, timeout, peak memory, OpenVINO version before the run, key package versions, environment id, failure id. |
| `failures[]` | One per `failed`/`timeout`/`error` result: category, heuristic error class, signature, failing cell index and source excerpt, exception type and message, traceback excerpt, log tail, links (job, test step, logs artifact, log file, notebook source). |
| `error_groups[]` | Failures grouped by signature, with affected notebooks and configs. |
| `summary` | Per-run metrics (see below). |

`environments.json` maps `env_id` to the full `pip freeze` (package → version or URL) of the notebook environment after execution.

### Statuses

| Status | Meaning |
|---|---|
| `passed` | treon exited with code 0. |
| `failed` | treon exited with a non-zero code (cell error, dead kernel, ...). |
| `timeout` | The notebook was killed after the per-notebook timeout. |
| `error` | Harness problem for this notebook (e.g. venv cloning failed, patched notebook missing). |
| `skipped` | Intentionally not run for this config: `skip_reason` is `skip_config` (`.ci/skipped_notebooks.yml`) or `ignore_list` (e.g. `.ci/tensorflow.txt`). |
| `not_run` | Planned but never executed: `not_run_reason` is `job_failed_before_tests:<step>`, `job_cancelled`, `job_skipped`, `job_interrupted` or `no_fragment`. |

Batches are reshuffled every night, so history must be keyed by **(`notebook`, `config_id`)**, never by batch or job.

### Failure classification

* `category` (from the log): `cell_error`, `kernel_died`, `timeout`, `harness_error`, `unknown`.
* `error_class` (heuristic, for triage): `timeout`, `harness`, `disk`, `memory`, `crash`, `dependency`, `network`, `openvino`, `assertion`, `other`.
* `signature`: 12-hex hash of category, exception type and the exception message with paths, numbers, hashes and addresses masked.
  The same signature across notebooks points to a shared cause (e.g. a broken dependency); across nights it identifies a recurring error.

## Per-run metrics (`summary`)

| Metric | Definition |
|---|---|
| `results` | Counts by status; `executed = passed + failed + timeout + error`; `planned` = all rows. |
| `pass_rate` | passed / executed |
| `notebook_pass_rate` | notebooks whose executed results all passed / notebooks with at least one executed result |
| `coverage` | executed / (planned − skipped) — share of applicable tests that actually ran (infrastructure health) |
| `matrix_coverage` | executed / (catalog size × number of configs) |
| `notebooks` | total, tested, untested (+ list), passing everywhere, failing somewhere (+ list), failing everywhere |
| `platform_specific_failures` | notebooks failing only on one OS / one Python version while passing on others |
| `by_config`, `by_os`, `by_python` | counts, pass rate, notebook time (`test_s`) and job time (`compute_s`) |
| `infra` | jobs total/succeeded/failed/cancelled, jobs failed before tests, notebooks lost (`not_run`), `infra_failure_rate` |
| `durations` | run wall clock, job time, notebook time, setup overhead, notebook median/p90/max, 10 slowest, results near the timeout (≥ 80 %) |
| `failures` | counts by category and error class, top error groups |
| `openvino_versions`, `openvino_dev_share` | OpenVINO versions notebooks actually ran with, share of dev/nightly builds |

## Cross-run metrics (computed by the dashboard)

Use only `run.event == "schedule"` runs of the `latest` branch for trends, deduplicated by `run.id` (highest `run.attempt`).
Key results by (`notebook`, `config_id`); "executed" means status in `passed|failed|timeout|error` (`skipped`/`not_run` nights
neither break nor extend a streak).

| Metric | Definition |
|---|---|
| New failure (regression) | failing now, last executed result `passed` |
| Fixed | `passed` now, last executed result failing |
| Failing since / streak | first run of the current consecutive failing streak (date, run, commit) and its length in nights; last passing run and commit; suspect range `https://github.com/openvinotoolkit/openvino_notebooks/compare/<last_pass_sha>...<first_fail_sha>` |
| Probable cause (at first failure) | `notebook_changed` (`notebooks[].dir_tree_sha` differs from the last pass), `openvino_changed` (`packages.openvino`), `packages_changed` (other key packages or `env_id`), otherwise `external_or_flaky` |
| Flakiness | pass↔fail transitions / (executed nights − 1) over the last N nights (default 14); flaky if ≥ 0.3 while `dir_tree_sha` is unchanged |
| Duration regression | `duration_s` > 1.5 × median of the last 7 passed durations and > +60 s |
| Timeout risk | median duration of the last 7 executed results ≥ 0.8 × `timeout_s` |
| MTTR | mean length of failure streaks closed within the window |
| Recurring errors | signatures seen on ≥ k nights, with first/last seen |
| Trends | `pass_rate`, `coverage`, `infra.jobs_failed_before_tests`, untested notebooks, `durations.compute_s`, per-notebook stability (pass rate over the window) |

## Local usage

```bash
# Fragments are produced by validate_notebooks.py when --dashboard_dir is passed.
python .ci/split_notebooks.py --num_batches 3 --seed 1 --plan_file plan/plan.json
python .ci/nightly_report/build_report.py --fragments_dir <downloaded fragments> --plan plan/plan.json \
    --output_dir out --ignore_list .ci/tensorflow.txt --workflow_file .github/workflows/treon_nightly.yml \
    --repository openvinotoolkit/openvino_notebooks --run_id <id>
# Without network access use --no_api (or --run_json/--jobs_json/--artifacts_json with saved API responses).
python -m pytest .ci/nightly_report/tests
```
