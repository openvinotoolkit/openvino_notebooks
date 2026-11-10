# Notebook review summary template

Replace placeholders with evidence-backed content. Keep applicable sections and
use `Not applicable` or `Unverified` explicitly. Repeat the model/task row for
each new or materially changed pairing. If findings are posted inline, reference
them rather than duplicating their full text. If there are no actionable findings,
say so and omit the empty findings table.

```markdown
## Submission review

- **Verdict:** <Ready for maintainer review / Changes recommended / Incomplete - verification needed>
- **Reviewed head:** `<head SHA>`
- **Scope:** <notebooks and related files reviewed>
- **Summary:** <brief description of the submission>

## Model/task suitability and usefulness

| Model / task | Suitability, added value, and evidence | Practicality, licensing, and responsible use | Assessment |
|---|---|---|---|
| <Model and task, or Not applicable> | <Evidence-backed assessment> | <Evidence or limitations> | <appropriate / needs justification / inappropriate / not applicable> |

## Findings

<No actionable findings, or reference verified inline findings, highest impact first.>

| Severity | Location | Evidence and impact | CONTRIBUTING.md guidance | Recommended action |
|---|---|---|---|---|
| <blocker / major / minor> | <immutable file link and line or notebook cell> | <specific evidence and user impact> | <section> | <specific fix> |

## Contribution checklist

| Area | Status | Evidence or limitation |
|---|---|---|
| Correctness | <Pass / Fail / Not applicable / Unverified> | <evidence> |
| Environment and portability | <Pass / Fail / Not applicable / Unverified> | <evidence> |
| Runtime and resources | <Pass / Fail / Not applicable / Unverified> | <evidence> |
| Documentation | <Pass / Fail / Not applicable / Unverified> | <evidence> |
| Metadata and integration | <Pass / Fail / Not applicable / Unverified> | <evidence> |
| Licensing and responsible use | <Pass / Fail / Not applicable / Unverified> | <evidence> |
| CI | <Pass / Fail / Not applicable / Unverified> | <evidence for the reviewed head SHA> |

## Validation and limitations

- **CI:** <results for the reviewed head SHA, or unavailable>
- **Inspected:** <files, cells, contribution-guide revision, primary sources checked>
- **Not executed:** Submitted notebook code and tests were not run during this review.
- **Unverified:** <claims or sources that could not be verified, or None>
- **Maintainer follow-up:** <focused question/action, or None>
```
