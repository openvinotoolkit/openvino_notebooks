# Project checklist for notebook reviews

Repository-specific rules that can be checked by reading the diff, the notebook
JSON and the `.ci` scripts. Apply only the sections relevant to the change. Each
rule names its usual severity; adjust it to the actual impact.

## Release-branch showcase submissions

Some teams submit notebooks to a release branch (a version name such as `2026.3`
or `2026.3.1`) instead of `latest`. Such a notebook shows a model to customers,
possibly one that is no longer current; it is not maintained after merge and not
tested in CI. Recognize it by the PR base branch and adjust the review:

- Do not request CI-only work: `test_replace` metadata, tiny models, CI runtime
  limits, `.ci/*.txt` or `.ci/skipped_notebooks.yml` entries. Skip the CI patching
  section below. Flag CI list lines unrelated to the notebook as noise.
- Only notebook metadata validation runs on release branches; codecheck, spelling
  and treon target `latest`. Missing results prove nothing either way: read
  formatting and spelling only where they affect readers.
- An older or superseded model is not a defect by itself. Responsible-use and
  licensing rules apply unchanged.
- No one fixes the notebook after merge, so the first Run All must work. Each of
  these is a **blocker**:
  - It runs from a fresh clone in a clean environment (see
    [Fresh-clone inputs](#fresh-clone-inputs)). Checkpoints and datasets have a
    download step or a direct link with setup instructions.
  - Steps described as optional (dataset evaluation, quantization) are skipped
    when their flag is off, and later cells do not require their outputs.
  - The default device is CPU or AUTO, or picked from available devices; nothing
    requires a GPU or NPU before the device selector.
  - The tested environment is documented (OS, Python version, key package
    versions, device), and the pins install on that Python version.
- Model usage and the scenario get the most attention, because nothing else will
  catch their errors: apply the model usage section strictly.

## Fresh-clone inputs

A notebook often works only on its author's machine: it reads a file that was
never committed, outputs of another notebook, a sibling project checkout, a
dataset at a local path, or a model already in the author's cache. Trace every
input in the notebook **and its helper `.py` files**:

- Each file the code reads is committed in the PR, downloaded by an earlier cell
  (attachment URL, Hugging Face Hub, pinned release), or created by an earlier
  cell. A `FileNotFoundError` guard or a "run ... first" message needs such a
  producer.
- Paths stay inside the notebook folder, except the repository `utils`
  (`../../utils`). Reading `../<other notebook>/...`, a parent of the repository
  or a "first existing of several locations" search hides a missing input.
- No other notebook has to run first, and no hand-over directory (`artifacts/`)
  is filled by code that is not in the PR.
- Configuration lives in cells or widgets, not in environment variables or named
  conda environments; anything read from the environment has a working default.
- Packages are installed by the notebook: every third-party import is covered by
  its install cell (or the repository `utils`), not by the author's environment.

Ask the author for a validation statement in the PR description: the notebook
ran with Run All from a fresh clone of the PR branch, in a new virtual
environment, with empty model caches (for example `HF_HOME` pointing to an empty
directory); OS, Python, key package versions and device. For a release-branch
showcase, which CI never runs, a missing statement is **major**.

## Model usage

- Pre- and postprocessing match the original implementation: resize versus
  letterbox padding, color order, normalization, decoding, thresholds and
  task-specific logic (for example, a tracker also updated on frames without
  detections). A PyTorch versus OpenVINO comparison that shares the submitted
  preprocessing cannot catch these errors; compare with the model's reference
  code (**major**).
- The scenario uses the model for what it was trained for, on suitable inputs.
- An accuracy or quality check cannot report success when its limit is exceeded.
  It follows the benchmark's evaluation rules or calls itself a simplified
  comparison (**major**).
- Flags do what their comments say (`QUANTIZE = False` skips quantization even
  when no INT8 IR exists yet). Printed labels name the device actually used.
- The notebook either runs the workflow or is explicitly a manual guide with
  commands to run elsewhere; code cells that do neither are a finding.
- Gated models: show how to log in (`whoami()`, then `notebook_login()` on error).

## Assets and README

- Images, videos and data samples are referenced by PR attachment URLs as
  CONTRIBUTING describes, not committed and not downloaded from the contributor's
  fork. Sample media states its source and license (**major** for committed
  binaries or media without a license).
- README matches the notebook: file names, clone and setup instructions, model and
  results. Images in it exist before the notebook is run. The Scarf pixel path
  names this notebook's README.
- No unrelated changes (other notebooks' files or CI list entries).

## CI patching

Not required for release-branch showcase submissions. CI runs a copy patched by
`.ci/patch_notebooks.py`:

- Every `metadata.test_replace` key must occur verbatim in its cell. A missing key
  raises `ValueError` and the notebook fails on every CI job (**blocker**). Black
  reformatting or a later edit of the cell often breaks the match.
- Every cell that downloads, exports, loads or generates with a full model must be
  bounded by a replacement: model id -> a tiny random model with the same
  `model_type` (say when its layer types differ from the real checkpoint, e.g. a
  hybrid tiny model for a dense model, because CI then covers another path);
  INT4 -> `int8` for tiny models (too few channels for INT4); fewer new tokens,
  steps, frames or a smaller resolution. A replacement must keep the code path
  executable, never turn it into `pass` or a no-op (**major**).
- Every cell containing both `gradio` and `launch(` is emptied. Definitions used
  by later cells must not live in the launch cell (**major**).
- `.ci/skipped_notebooks.yml`: the `os`, `python` and `device` keys of one skip
  entry are OR-ed, so `{os: [windows-2022], device: [gpu]}` skips every Windows
  run and every GPU run. Each new skip needs a stated reason (**minor**, **major**
  when it hides the platform the notebook targets).

## Dependencies

- A dependency on unreleased support (`git+...` of Optimum Intel or another
  library, the OpenVINO nightly index) must be pinned to a commit, not a branch.
  Report the state of the upstream PR (open or merged) and the release that should
  contain it; the notebook text should say what to replace after that release. An
  unpinned moving branch is **major**.
- Do not install a package twice from different channels (for example stable
  OpenVINO, then nightly); `openvino`, `openvino-tokenizers` and `openvino-genai`
  must come from the same release (**minor**, **major** when versions can mismatch).
- Version bounds must agree with the libraries that use them (e.g. the
  `transformers` range required by the pinned Optimum Intel). CONTRIBUTING asks for
  the widest compatible range: an exact pin needs a reason (unreleased support, a
  known breaking release).
- PyTorch for export comes from the CPU index unless GPU PyTorch is required.

## Saved outputs

Committed outputs are for readers, not proof of a run. Useful outputs show model
results: generated text, predictions, small images or plots. Missing outputs make
runtime claims `Unverified`; they are not a defect by themselves. Proof of a clean
run is the author's validation statement (see
[Fresh-clone inputs](#fresh-clone-inputs)), checked against the outputs that are
present. Request cleanup (**minor**, **major** when it exposes internal
information) of:

- installation logs (`pip` output, "restart the kernel" notes), download
  progress bars and other setup noise;
- local or internal paths, user names, host names, internal URLs or tokens;
- large images, embedded videos or animations that inflate the notebook; show
  them in the PR description or keep them small.

When outputs exist, check:

- They were produced by the current code: execution order, no output of removed
  or renamed code, the model id and precision match the cells.
- They come from the real model, not the CI tiny model, and the device matches
  the claim.
- Generated text is not degenerate: empty, one unit repeated 30+ times, a tail
  that loops until the token limit, or an answer cut at the token limit.
- No tracebacks, failed downloads or warnings that change behavior.
- Execution counts of cells that keep them increase through the notebook. Counts
  such as `4, 15, 6` or a repeated count mean cells were run out of order or
  re-run, so the outputs may not match a clean Run All.

## LLM and VLM notebooks

- The chat template is applied once: no BOS added twice when templated text is
  tokenized again, no second template applied by the pipeline (`apply_chat_template`
  in GenAI) to an already templated prompt.
- Multi-turn chat and chat demos send the whole history through the chat template
  (Gradio 6 history content is a list of parts such as
  `[{"type": "text", "text": ...}]`). The answer limit lets typical answers end
  with EOS: a truncated answer stored in the history can make follow-up requests
  fail or come back empty (**major** for a demo).
- An empty answer or an input-limit error in a demo is shown to the user
  (`gr.Error`, a readable message), not raised as an unhandled exception.
- Tool calling parses model output defensively: invalid JSON from the model must
  not stop Run All.
- Device-specific settings (NPU static shapes or compression, cache directories)
  are applied only for the selected device.

## Gradio

- The local-first launch with a reported `share=True` fallback described in
  CONTRIBUTING is accepted. `share=True` on the first launch attempt is
  **major**; a fallback that does not print the local launch error is **minor**.
- User-facing launches use `debug=True`; the CI replacement or the emptied launch
  cell keeps CI from blocking.
- The remote-launch comment from CONTRIBUTING is kept.

## Severity

- **blocker:** CI patching fails, Run All breaks on a clean kernel, prohibited task
  or license, credentials or private data in the notebook.
- **major:** wrong, empty or degenerate results; a broken demo; an unbounded
  full-model path in CI; an unpinned unreleased dependency; missing required
  structure (README, Scarf pixel, telemetry, `setup_notebook_venv()`).
- **minor:** documentation gaps, style, redundant installs, cosmetic output.
- For a release-branch showcase, CI-only issues are not findings.
