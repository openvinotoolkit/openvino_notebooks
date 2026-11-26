## Summary

<One or two sentences: what fails in the nightly, for which notebooks, and since when (use the group's Trend/History).>

## Error signature

Error signature: `<12-hex-signature>`
Nightly run: <nightly run URL>
Related signatures: <comma-separated backtick-quoted covered signatures, or none>

## Affected notebooks

<Every (notebook, config) pair from the group's "Affected notebooks" table, not only the representative failure. For infrastructure groups, list the notebooks that were not run and the affected jobs.>

| Notebook | Configs | Status | Previous night |
|---|---|---|---|
| `notebooks/<dir>/<notebook>.ipynb` | `<config-id>`, `<config-id>` | <failed/not run> | <same error/passed/other failure/not run/n/a> |

## Error type and where it starts

- **Category / class:** <category> / <error class>
- **Exception:** `<ExceptionType>: <short message>`
- **Starts at:** `<notebook or helper path>`, cell <N> (`<first line of the failing code>`), or `<job/step>` for infrastructure failures; first seen <date or run>

```text
<Shortest traceback or job-log excerpt that shows the origin of the error>
```

## Evidence

- **Logs:** <representative job and log links>
- **Environment:** <relevant package versions and differences to a passing run, or "no relevant differences">
- **History:** <trend, nights with the same error, and OpenVINO versions observed failing or passing>
- **Related items:** <related issues/PRs found, or "none found">

## Investigation

<What was checked (files, searches, hypotheses) and what each check showed.>

## Why no automated fix

<The concrete reason: transient or infrastructure failure, OpenVINO runtime issue without a local cause, fix outside the allowed files, ambiguous evidence, failed self-check, or exhausted budget.>

## Suggested next step

<A concrete action for a maintainer, such as a specific file and change to try, an upstream component to report to, or a command to reproduce the failure.>

<details>
<summary>AI self-improvement</summary>

### Suggestions for future runs

- **Future-run improvements:** <One to three concrete, conditional improvements derived from this investigation and the suggested remedy or blocker: which evidence to collect first, detection/grouping signal to use, check to add or prioritize, or ineffective approach to avoid. Cite supporting evidence and state when each suggestion applies and what remains unverified. These are proposals for later runs, not permission to change workflow safeguards.>

### Before investigation

- **Prior feedback:** <Source PR/issue URLs or memory runs, observed outcomes and uncertainty; or none available with reason.>
- **Applied learning:** <The applicable lesson and how it changed this investigation, or why prior advice did not apply.>

### After investigation

- **Self-assessment:** <What the investigation established, ruled out or could not verify, including why no safe fix was possible.>
- **Reusable lesson:** <The highest-value future-run improvement to retain in memory, with its applicability condition, supporting evidence and limitations.>

</details>
