## Problem

<One or two sentences: what fails in the nightly, for which notebooks, and since when (use the group's Trend/History).>

## Error signature

Error signature: `<12-hex-signature>`
Nightly run: <nightly run URL>

## Affected notebooks

<Every (notebook, config) pair from the group's "Affected notebooks" table, not only the representative failure.>

| Notebook | Configs | Status | Previous night |
|---|---|---|---|
| `notebooks/<dir>/<notebook>.ipynb` | `<config-id>`, `<config-id>` | <failed/not run> | <same error/passed/other failure/not run/n/a> |

## Error type and where it starts

- **Category / class:** <category> / <error class>
- **Exception:** `<ExceptionType>: <short message>`
- **Starts at:** `<notebook or helper path>`, cell <N> (`<first line of the failing code>`), first seen <date or run>
- **Representative log:** <job or log link>

```text
<Shortest traceback excerpt that shows the origin of the error>
```

## Root cause

<The causal chain from evidence to cause, citing the log excerpt, package/environment differences, history, or source lines that support it.>

## Fix

<What was changed and why it addresses the root cause. Name each changed file and summarize the change; mention notebook formatting or dependency conventions that were preserved.>

## Verification and limits

| Check | Result |
|---|---|
| `<command>` | <passed/failed/not run, with reason> |

- Normal CI is not triggered automatically for PRs created by this workflow; maintainers must run the notebook checks before merging.
- <Anything not verified, such as the notebook not being executed end to end, hardware-specific configs, or remaining uncertainty.>

## History

<Trend and per-night summary from the group's History table, previous remediation attempts or related issues/PRs, and OpenVINO versions observed failing or passing.>
