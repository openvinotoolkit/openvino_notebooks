# Contributing to OpenVINO Notebooks

- [Contributing to OpenVINO Notebooks](#contributing-to-openvino-notebooks)
  - [Design Decisions](#design-decisions)
    - [General design considerations](#general-design-considerations)
    - [Implementation choices](#implementation-choices)
    - [Coding Guidelines](#coding-guidelines)
    - [Other things to keep in mind](#other-things-to-keep-in-mind)
    - [Notebook naming](#notebook-naming)
    - [`README.md` files](#readmemd-files)
    - [File Structure](#file-structure)
      - [Recommendations for File Structure](#recommendations-for-file-structure)
    - [Notebook utils](#notebook-utils)
    - [Interactive inference with Gradio](#interactive-inference-with-gradio)
    - [Notebooks Metadata](#notebooks-metadata)
  - [Requirements](#requirements)
  - [Validation](#validation)
    - [Automated tests](#automated-tests)
    - [Manual test and code quality tools](#manual-test-and-code-quality-tools)
      - [`treon`](#treon)
      - [`nbqa`](#nbqa)
      - [`nbdime`](#nbdime)
      - [JupyterLab Code Formatter](#jupyterlab-code-formatter)
      - [Black Automatic Code Formatter](#black-automatic-code-formatter)
      - [`PySpelling`](#pyspelling)
  - [Getting started](#getting-started)
    - [Pull Requests (PRs)](#pull-requests-prs)
  - [Help](#help)

Thank you for being interested in contributing to the OpenVINO Notebooks repository! This guide
explains the design decisions, requirements, and coding guidelines for the OpenVINO Notebooks
repository.  Please read the Design Decisions and Validation sections before jumping to the Getting
Started section.

The goal of this document is to make it as easy as possible to contribute to the OpenVINO Notebooks
repository, while maintaining the quality and consistency of the notebooks in the repository.

If you have a question, about the notebooks or about contributing to the repository, please create a
[discussion](https://github.com/openvinotoolkit/openvino_notebooks/issues)!

## Design Decisions

The goals of the OpenVINO Notebooks are:

- to make it easy to get started with OpenVINO.
- to teach how to use OpenVINO tools to do inference, convert and quantize models.
- to make it easy to use models from OpenVINO's Open Model Zoo and other public models.

To do this, there are a few requirements that all notebooks need to pass.

### General design considerations

1. The notebooks work on Windows, macOS and Linux (see [supported operating
   systems](https://github.com/openvinotoolkit/openvino_notebooks#%EF%B8%8F-system-requirements))
   with Python 3.9, 3.10, 3.11 and 3.12.
2. As a rule, the notebooks do not require installation of additional software that is not installable by
   `pip`. We do not assume that users have installed XCode Developer Tools, Visual C++ redistributable,
   `cmake`, etc. Please discuss if your notebook does need C++ - there are exceptions to this rule.
3. The notebooks should work on all computers, and in container images. We cannot assume that a
   user will have an iGPU or a webcam, so using these should be optional. For example, In the case
   of webcam inference, provide the option to use a video.
4. The notebooks should work in Jupyter Lab and Jupyter Notebook. If a dependency does not work in
   either of these, the notebooks should still be usable: use graceful degradation. For example, if
   a visualization library only works in Jupyter Lab, but offers substantial advantages, make sure
   that a user who runs Jupyter Notebook still sees the output, even if it is not
   interactive/3D/annotated/etc.
5. With the exception of notebooks that demonstrate training of neural networks, all notebooks
   should by default run in less than five minutes with "Run All Cells" (excluding time required to
   download files). If this means using a smaller model or smaller dataset that gives less than
   optimal results, or having a less amazing visualization, provide the better option that takes
   longer as an option.
6. Not everyone who uses the notebooks will have a fast computer and/or fast internet. It is not
   always possible to use a smaller model or a smaller dataset, but if it is, please do that, and
   provide an option for the larger model or dataset.
7. The target audience for the notebooks includes both experienced and new developers. The goal is
   not just to show the output of a model, but to teach how OpenVINO works, by interacting with it.
   Not all notebooks need to be full-fledged tutorials, but it is always good to explain steps and
   add comments.
8. Respect for human rights is rooted in our [values at Intel](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). 
   We will not accept contributions that perform facial recognition or analyze demographics like age
   and gender. 

### Implementation choices

1. Notebooks in this repository typically rely on a shared `requirements.txt` file. 
   However, contributors are encouraged to install the required packages at the top of their notebook using 
   `%pip install -q ...` commands. This allows the notebooks to be run independently as standalone examples. 
   To maintain package compatibility, contributors are expected to install the same versions of packages 
   as specified in the shared `requirements.txt` file. This helps ensure consistency in our testing pipelines 
   and prevents dependency conflicts.
2. The notebooks are located in the "notebooks" subdirectory. There is a subdirectory for every
   notebook, with generally the same base name as the notebook. For example, the
   `hello-world.ipynb` notebook can be found in the hello-world directory.
   - See the [Notebook naming](#notebook-naming) section below, for the
     naming of the notebooks.
   - Add a `README.md` to the notebook subdirectory. Add a screenshot that gives an indication of what
     the notebook does if applicable.
   - Avoid adding any other files to the notebook's subdirectory. Instead, rely on models and data samples available online and fetch them within the notebook. Please refer to the [Notebook utils](#notebook-utils) section.
3. In case you want to utilize one of the Open Model Zoo models, refer to the [Model Tools](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/model-tools)
   notebook.
4. The notebooks should provide an easy way to clean up the downloaded data, for example with a
   commented-out cell at the end of the notebook.

### Coding Guidelines

1. See [PEP 20](https://www.python.org/dev/peps/pep-0020/)
2. Format notebook code with [Black](https://github.com/psf/black), with a line width of 160. 
   See [Tools](#manual-test-and-code-quality-tools).
3. Use f-strings for string formatting: [PEP 498](https://www.python.org/dev/peps/pep-0498/)
4. Use keyword/named arguments when calling a function with more than one parameter:
   `function(a=1, b=2)` instead of `function(1, 2)`
5. Use `from pathlib import Path` for path manipulation instead of `os.path`
6. Add type hints to functions: [PEP 484](https://www.python.org/dev/peps/pep-0484/)
7. Add REST style docstring (see [action-recognition-webcam](./notebooks/action-recognition-webcam/action-recognition-webcam.ipynb)
   for an example). It is not necessary to specify the parameter type in the docstring, since
   type hints are already added to the function definition.
8. Do not use global variables in functions: a function should not depend on values that are
   defined outside it.
9. Use ALL_CAPS for constants.
10. Prefer consistency. Example: if other notebooks use `import numpy as np` do not use
   `import numpy` in yours.

### Other things to keep in mind

1. Always provide links to sources. If your notebook implements a model, link to the research paper
   and the source GitHub (if available).
2. Use only data and models with permissive licenses that allow for commercial use, and make sure to
   adhere to the terms of the license.
3. If you include code from external sources in your notebook add the
   name, URL and license of the third party code to the `licensing/third-party-programs.txt` file.
4. Don't use HTML for text cells, use Markdown markups instead.
5. Add **Table of content** to top of the Notebook, it helps to get quick fist understanding of content and ease of navigation in the dev environment. There is no need to think about it during development, it can be built or updated after changes with `.ci\table_of_content.py`. Just run the script with the parameter `-s/--source`, specifying a Notebook or a folder with several notebooks as value, the changes will be applied to all of them.
6. Add **Installation Instructions** section to the top of the notebook (after "Table of content") and to the corresponding `README.md` file in the notebook directory. See existing notebooks for the reference.
7. Add Scarf Pixel tag for analytics to the notebook (at the end of the first cell) and to the corresponding `README.md` file (to the end of the file). Add relative path to the notebook or `README.md` file as `path` URL query parameter. Example: `<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=<RELATIVE_FILE_PATH>" />`. You can use the following command to generate the tag and add it to the file: `python .ci/scarf_pixel.py -s <PATH>`.
8. Add telemetry snippet to the notebook file (e.g. to the cell with fetching notebook_utils file) for tracking notebook execution. Add notebook file name as a parameter to `collect_telemetry()` function. You can use the following command to generate the snippet and add it to the notebook file: `python .ci/telemetry_snippet.py -s <PATH>`. Snippet example:
   ```python
   # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
   from notebook_utils import collect_telemetry
   collect_telemetry("<NOTEBOOK_NAME>.ipynb")
   ```
1.  In case if notebook has specific requirements on python version or OS, it should be noted on top of notebook (before any code blocks) using
   following colored block:
   ```
   <div class="alert alert-block alert-danger"> <b>Important note:</b> This notebook requires python >= 3.9. Please make sure that your environment fulfill to this requirement  before running it </div>
   ```

### Notebook naming

 - Names should be descriptive but not too long.
 - Please use hyphen symbol `-` (not underscore `_`) to separate words in directories and notebooks names.
 - Directory name should match the corresponding notebook file name (e.g. `./hello-word/hello-word.ipynb`).

### `README.md` files

Every notebook must have a `README.md` file that briefly describes the content of the notebook. A simple structure for the `README.md` file is described below:

``` markdown
# Title of Tutorial
[brief intro, basic information about what will be described]

## Notebook Contents
[more details, possibly information about research papers, the model(s) used and/or data]
Additional subsections, e.g license information.


## Installation Instructions
[link to installation guide, other important information for install process]
```
Notebooks that work in Binder have a _Launch Binder_ badge in the `README.md` files.
In the same way, notebooks that work in Google Colab have a _Launch Colab_ badge in the `README.md` files.


### File Structure

To maintain consistency between notebooks, please follow the directory structure outlined below.

```markdown
notebooks/
└──<title>/
   ├── README.md
   ├── <title>.ipynb
   ├── utils/
   ├── model/
   ├── data/
   └── output/
```

In case the example requires saving additional files to disk (e.g. models, data samples, utility modules, or outputs),
please create corresponding folders on the same level as `README.md` file.

#### Recommendations for File Structure

- Model

We recommend to load your models using URL or other ways of distribution of pre-trained models, 
like PyTorch Hub or the Diffusers package. 

- Data

We recommend to use embedded URL for image/video data.
Follow the below instructions to create embedded URL in GitHub:
  - Go to any issue on GitHub.
  - In the comment section, you can attach files. Just drag/drop, select or paste your image.
  - Copy the code/link displayed in the text area

- License

If you download or include a model, it must be licensed under an open source license like Apache 2.0 which allows for redistribution, modification and commercial use. 

Any datasets, images or videos used for fine-tuning, quantization or inference inside a notebook must be licensed under Creative Commons 4.0 (CC 4.0) with permission for commercial use. If commercial use is not allowed, but the data is under CC 4.0, special approval will be required. Please let us know in your pull request if your data has any restrictions on use.

### Notebook utils

The `notebook_utils.py` file in the `./utils` directory (in the repository root) contains utility functions and classes that can be reused across
notebooks. It contains a `download_file()` function that optionally shows a progress bar, and a standard way to convert
segmentation maps to images and display them.


### Interactive inference with Gradio
To enhance the functionality of your notebook, it is recommended to include an interactive model inference interface at the end. We recommend using [Gradio](https://gradio.app) for this purpose.

Here are some guidelines to follow:

 - Install the latest version of Gradio by running `pip install -q -U gradio`.
 - If you're using a `gradio.Interface` object in your demo, disable flagging by setting the `allow_flagging` keyword argument to `'never'`.
 - Launch the interface with `debug=True`. This mode blocks the main thread, identifies any possible inference errors and stops the running Gradio server when execution is interrupted. It is important to disable it when executing on CI to prevent main thread hanging. It is done by adding `test_replace` metadata key to the cell containing the line with `launch` method, and replacing this line with the same one but excluding `debug` argument. More detailed instructions can be found [here](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Notebooks-Development---CI-Test-Speedup).
 - Avoid setting `share=True` for the `launch` method of `gradio.Interface`. Enabling this option generates an unnecessary public link to your device, which can pose a security risk. Instead, if the interface window is not rendering in your case, consider temporarily setting the `server_name` and `server_port` parameters to the address where the server is located. This workaround is particularly useful when you are using remote Jupyter server. To assist other users, please leave a comment in your notebook explaining this solution. It will help them quickly resolve the issue if they encounter the same problem. The comment we recommend to use for this:

```
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/
```

- We use Gradio Blocks only when we need to create a complex interface. The Gradio Interface class provides an easy-to-use interface and saves development time, so we use it whenever possible. However, for more complex interfaces, Gradio Blocks gives us more flexibility and control.


### Notebooks Metadata

Each notebook file has metadata that includes additional information about the notebook. Some metadata fields (e.g. title, creation date, links to GitHub, Colab, Binder etc.) are generated automatically from notebook content or related `README.md` file. However other fields (e.g. tags, image URL) should be defined by notebook contributor. As each notebook file has JSON format, manually defined metadata fields are stored in corresponding `.ipynb` file in global notebook metadata object (`metadata.openvino_notebooks` field in the end of notebook JSON structure).

Example of such manually defined notebook metadata:

```JSON
"openvino_notebooks": {
 "imageUrl": "...",
 "tags": {
  "categories": [
   "First Steps"
  ],
  "libraries": [],
  "other": [],
  "tasks": [
   "Image Classification"
  ]
 }
}
```

Notebook tags in metadata can have several values and should be a subset of defined tags that can be found in `./selector/src/shared/notebook-tags.js`.
 - `tags.categories` tags relate to notebook groups like "AI Trends", "First Steps", "Model Demos" etc.
 - `tags.tasks` tags relate to particular AI tasks that are demonstrated in notebook.
 - `tags.other` tags are free-form tags and can be any string (please follow capitalization naming convention).


## Requirements

Contributors are encouraged to install the required packages at the top of their notebook using 
`%pip install ...` commands. This allows the notebooks to be run independently as standalone examples. 
To maintain package compatibility, contributors are expected to install the same versions of packages 
as specified in the shared `requirements.txt` file located in the repository root folder.
Additional guidelines:
1. Specify the widest compatible package version range. If your notebook has only a lower bound on some package version, consider specifying it with ">=" sign instead of "==". Specifying the exact version of package might lead to dependency conflict between notebooks. 
2. Do not use spaces between package, version and comparison operator when specifying the package installed. Use "package==version" instead of "package == version".

## Validation

### Automated tests

We use GitHub Actions to automatically validate that all notebooks work. The following tests run automatically on a new notebook PR:

- `treon`: tests that the notebooks execute without problems on all supported platforms. 
- Code check: 
  - Uses [`flake8`](https://github.com/pycqa/flake8) to check for unnecessary imports and variables 
and some style issues
  - Verifies that the notebook is included in the main `README.md` and the `README.md` in the notebooks directory. 
  - Runs the `check_install.py` script to test for installation issues
- Spell check: spell checking is performed by `PySpelling` module which requires `Aspell` spell checker in conjunction with our custom word list dictionary (`.ci/spellcheck/.pyspelling.wordlist.txt`). For information about dealing with found spelling problems please refer to the [`PySpelling` section below](#pyspelling).
- `docker_treon`: tests that the docker image builds, and that the notebooks execute without errors in the Docker image. 
  To manually run this test, build the Docker image with `docker build -t openvino_notebooks .` and run the tests with
  `docker run -it  --entrypoint /tmp/scripts/test openvino_notebooks`. It is recommended to build the image on a clean 
  repository because the full notebooks folder will be copied to the image.
- [`CodeQL`](https://codeql.github.com/)
- Notebooks Metadata Validation: verifies that all added or modified notebooks in PR have valid metadata and visualizes them in workflow summary.

  - In the rest of this guide, the automated tests in GitHub
Actions will be referred to as CI (for Continuous Integration).

If your notebook takes longer than a few minutes to execute, it may be possible to patch it in the CI, to make 
it execute faster. As an example, if your notebook trains for 20 epochs, you can set it to train for
1 epoch in the CI. If you do inference on 100 frames of a video, you can set it to do inference on only 1. See 
[this Wiki page](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Notebooks-Development---CI-Test-Speedup) for more information.

In CI notebooks are validated using the `.ci/validate_notebooks.py` script, which uses [`treon`](#treon) for notebooks execution. 
The script prepares lists of testing and ignored notebooks based on the several arguments: `--os`, `--python`, `--device` or `--ignore_list`.
If `--os`, `--python` or `--device` arguments are provided, the script looks through `.ci/skipped_notebooks.yml` file to get ignored notebooks list.
Providing `--ignore_list` argument with `*.txt` files you can extend the ignored notebooks list.

To skip validation of a particular notebook, you should modify the existing notebook entry or add a new one to the `.ci/skipped_notebooks.yml` and define list of skip configurations as `yaml` objects with one of `os`, `python`, `device` keys or their combinations.


### Manual test and code quality tools

See [Getting started](#getting-started) about installing the tools mentioned in this section.

#### `treon`

Tests are run in the CI with [`treon`](https://pypi.org/project/treon/), a test framework for Jupyter Notebooks.

To run `treon` locally, run `treon` to run the tests for all notebooks, or `treon notebook.ipynb` for just one notebook. `treon` fails if the notebook environment is not
`openvino_env`.


#### `nbqa`

[`nbqa`](https://github.com/nbQA-dev/nbQA) allows using a variety of code quality tools on Jupyter
Notebooks. For example `nbqa flake8 notebook.ipynb` will warn about unused imports.

#### `nbdime`

[`nbdime`](https://github.com/jupyter/nbdime) has several useful tools, among which `nbdiff-web` to
show the difference between two notebooks in a web browser. `nbdiff` can also be used as the
standard `diff` tool for `git`, with much more useful output than the regular `git diff` output.

#### JupyterLab Code Formatter

[JupyterLab Code Formatter](https://ryantam626.github.io/jupyterlab_code_formatter/index.html) adds a
button to Jupyter Lab to automatically format the code in notebooks with `black` and `isort`. Please
use either this extension or a different way to automatically format your notebook.

#### Black Automatic Code Formatter
[Black](https://black.readthedocs.io/en/stable/getting_started.html) is the uncompromising Python code formatter that has extension for Jupyter notebooks.

Install `black` with Jupyter Notebooks support:
```
python3 -m pip install black[jupyter]
```
Run formatting command in notebooks directory:
```
black -l 160 <notebooks_dir>
```

#### `PySpelling`

[`PySpelling`](https://facelessuser.github.io/pyspelling/) is a module to help with automating spell checking and it is essentially a wrapper around the `Aspell` command line utility. Additional custom (project and domain specific) word list dictionary that extends standard `Aspell` dictionary is located in `.ci/spellcheck/.pyspelling.wordlist.txt` file. `PySpelling` configuration file can be found in `.ci/spellcheck/.pyspelling.yml` file.

To run spell checking locally, execute the following command:
```
python .ci/spellcheck/run_spellcheck.py
```

If spell check is failing, there are any typos or new words, you have two possible options how to fix it:

1. Add new word (abbreviation, name, term etc.) to the word list dictionary (`.ci/spellcheck/.pyspelling.wordlist.txt`)
2. Skip single occurrence of unknown word or even whole phrase - just wrap the text with `<spell>` tag (for example, `<spell>Unknown word or phrase</spell>`). Note that `<spell>` is a custom tag and it doesn't affect the Markdown formatting and style (unlike <spell>backticks</spell> in preformatted text and code blocks).

## Getting started

1. Create a fork, a copy of the repository, by clicking on the Fork button on the top right of the
   OpenVINO Notebooks [GitHub page](https://github.com/openvinotoolkit/openvino_notebooks)
2. Install the recommended packages for a development environment with `pip install -r
   .ci/dev-requirements.txt` inside the `openvino_env` environment. This installs all the packages
   mentioned in the [Validation](#Validation) section.
      > **Note**: `PySpelling` dependency from `.ci/dev-requirements.txt` requires `Aspell` for spell checking that should be installed manually. For installation instructions please refer to the [`PySpelling` documentation](https://facelessuser.github.io/pyspelling/#prerequisites).
3. Create a branch in this fork, from the *latest* branch. Name the
   branch however you like.
4. Double-check the points in the [Design Decisions](#design-decisions) and [Validation](#Validation) sections.
5. Check that your notebook works in the CI
   - Go to the GitHub page of your fork, click on _Actions_, select `treon` on the left. There will
     be a message _This workflow has a workflow_dispatch event trigger._ and a _Run workflow_ button.
     Click on the button and select the branch that you want to test.
6. Test if the notebook works in [Binder](https://mybinder.org/) and [Google Colab](https://colab.research.google.com/) and if so, add _Launch Binder_ and _Launch Colab_ badges 
   to the `README.md` files.

Once your notebook passes in the CI and you have verified that everything looks good, make a Pull Request!

### Pull Requests (PRs)

1. If some time has passed since you made the fork, sync your fork via GitHub UI or update your form manually - rebase or merge `openvinotoolkit/openvino_notebooks` repository *latest* branch to the *latest* branch in your fork.
2. Create your PR against the `openvinotoolkit/openvino_notebooks` repository *latest* branch.
3. Please create a description of what the notebook does with your PR. Screenshots are appreciated!
4. On making or updating a Pull Request, the tests in the CI will run again. Please keep an
   eye on them. If the tests fail and you think the issue is not related to your PR, please make a comment on your PR.

## Help

If you need help at any time, please open a
[discussion](https://github.com/openvinotoolkit/openvino_notebooks/issues)! If you think one of the
guidelines is too strict, or should not apply to you, feel free to ask about that too.
