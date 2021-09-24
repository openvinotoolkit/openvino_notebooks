# Contributing to OpenVINO Notebooks

- [Contributing to OpenVINO Notebooks](#contributing-to-openvino-notebooks)
  - [Design Decisions](#design-decisions)
    - [General design considerations](#general-design-considerations)
    - [Implementation choices](#implementation-choices)
    - [Other things to keep in mind](#other-things-to-keep-in-mind)
    - [Notebook naming](#notebook-naming)
    - [Readmes](#readmes)
    - [File structure](#file-structure)
  - [Validation](#validation)
    - [Automated tests](#automated-tests)
    - [Manual test and code quality tools](#manual-test-and-code-quality-tools)
      - [nbval](#nbval)
      - [nbqa](#nbqa)
      - [nbdime](#nbdime)
      - [JupyterLab Code Formatter](#jupyterlab-code-formatter)
  - [Getting started](#getting-started)
    - [Pull Requests (PRs)](#pull-requests-prs)
  - [Help!](#help)

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
   with Python 3.6, 3.7 and 3.8.
2. The notebooks do not require installation of additional software that is not installable by
   `pip`. We do not assume that users have installed XCode Dev Tools, Visual C++ redistributable,
   cmake, etc.
3. The notebooks should work on all computers, and  in container images. We cannot assume that a
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

### Implementation choices

1. The notebooks use one shared requirements.txt. If "the notebooks don't work" it is often caused
   by a dependency of a dependency having an issue. We are therefore reluctant to add new
   dependencies and will only add them if they add real value. Do not let this discourage you if
   you do want to include a certain package! If it is necessary, or can be useful for other
   notebooks too, we are open to adding it.
2. All notebooks are saved with the `openvino_env` kernel. This ensures that there is never a
   conflict between a user's other Python installations and the notebook installation.
3. Notebook code is automatically formatted with [Black](https://github.com/psf/black), with a line
   width of 100. We did not choose `black` because black offers the best or nicest formatting, but
   because consistency is more important than preferences, and that time spent on prettifying code
   is time not spent on other useful things.
4. Imports are at the top of the notebooks file, sorted alphabetically with `isort`, grouped
   according to [PEP 8](https://pep8.org/#imports)
5. The notebooks are located in the "notebooks" subdirectory. There is a subdirectory for every
   notebook, with generally the same base name as the notebook.  For example, the
   001-hello-world.ipynb notebook can be found in the 001-hello-world directory.
   - See the [Notebook naming](#notebook-naming) section below, for the
     numbering of the notebooks.
   - Add a README to the notebook subdirectory. Add a screenshot that gives an indication of what
     the notebook does if applicable.
   - Add any supporting files to this subdirectory too. Supporting files should
     be small (generally less than 5MB). Larger images, datasets and model
     files should be downloaded from within the notebook.
6. All related files, with the exception of Open Model Zoo models, should be saved to the notebook subdirectory, 
   even if that means that there is a small amount of duplication. For Open Model Zoo models, see the directory
   structure in the [104 Model Tools](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/104-model-tools) 
   notebook.
7. The notebooks should provide an easy way to clean up the downloaded data, for example with a
   commented-out cell at the end of the notebook.

### Other things to keep in mind

1. Always provide links to sources. If your notebook implements a model, link to the research paper
   and the source Github (if available).
2. Only use data and models that have a license that permits usage for commercial purposes.

### Notebook naming

Names should be descriptive but not too long. We use the following numbering scheme:

- `000-` hello world like notebooks: very small tutorials that help to quickly show how OpenVINO works.
- `100-` OpenVINO tool tutorials: explain how to optimize and quantize notebooks.
- `200-` OpenVINO model demos: demonstrate inference on a particular model.
- `300-` Training notebooks: notebooks that include code to train neural networks.
- `400-` Live demo notebooks: demonstrate inference on a live webcam.

### READMEs

Every notebook must have a README file that briefly describes the content of the notebook. A simple structure for the README file is described below:

``` markdown
# Title of Tutorial
[brief intro, basic information about what will be described]

## Notebook Contents
[more details, possibly information about research papers, the model(s) used and/or data]
Additional subsections, e.g license information.


## Installation Instructions
[link to installation guide, other important information for install process]
```

Every notebook is also added to the notebooks overview table in the main
[README](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md) and the 
[README](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/README.md) in the notebooks directory
Notebooks that work in Binder have a _Launch Binder_ badge in the README files.


### File structure

To maintain consistency between notebooks, please follow the directory structure outlined below.

```markdown
<three-digit-number>-<title>/
├── README.md
├── <three-digit-number>-<title>.ipynb
├── utils/
├── model/
└── data/
```

In case of output provided by Notebook please create folder ```output``` on the same level as readme file.

## Validation

### Automated Tests

We use Github Actions to automatically validate that all notebooks work. The automated tests
currently test that the notebooks execute without problems on all supported platform. More granular
tests are planned. In the rest of this guide, the automated tests in Github Actions will be
referred to as CI (for Continuous Integration).

### Manual test and code quality tools

See [Getting started](#getting-started) about installing the tools mentioned in this section.

#### nbval

Tests are run in the CI with [nbval](https://github.com/computationalmodelling/nbval), a plugin for
py.test. The tests will only pass if the output is stripped from the notebooks. There are different
way to do this. `jupyter nbconvert --clear-output --inplace notebook.ipynb` should work without
installing additional dependencies. It is also possible to add a pre-commit hook to do this. The
article [Making Git and Jupyter Notebooks play
nice](http://timstaley.co.uk/posts/making-git-and-jupyter-notebooks-play-nice/) offers more
information and possible solutions.

To run nbval locally, run `pytest --nbval .` to run the tests for all notebooks, or `pytest --nbval
notebook.ipynb` for just one notebook. `nbval` fails if the notebook environment is not
`openvino_env`.

#### nbqa

[nbqa](https://github.com/nbQA-dev/nbQA) allows using a variety of code quality tools on Jupyter
Notebooks. For example `nbqa flake8 notebook.ipynb` will warn about unused imports.

#### nbdime

[nbdime](https://github.com/jupyter/nbdime) has several useful tools, among which `nbdiff-web` to
show the difference between two notebooks in a web browser. `nbdiff` can also be used as the
standard `diff` tool for `git`, with much more useful output than the regular `git diff` output.

#### JupyterLab Code Formatter

[JupyterLab Code Formatter](https://jupyterlab-code-formatter.readthedocs.io/en/latest/) adds a
button to Jupyter Lab to automatically format the code in notebooks with black and isort.

## Getting started

1. Create a fork, a copy of the repository, by clicking on the Fork button on the top right of the
   OpenVINO Notebooks [Github page](https://github.com/openvinotoolkit/openvino_notebooks)
2. Install the recommended packages for a development environment with `pip install -r
   .ci/dev-requirements.txt` inside the `openvino_env` enviroment. This installs all the packages
   mentioned in the [Validation](#Validation) section.
3. Create a branch in this fork, from the *main* branch. Name the
   branch however you like.
4. Doublecheck the points in the [Design](#Design) and [Validation](#Validation) sections.
5. Check that your notebook works in the CI
   - Go to the GitHub page of your fork, click on _Actions_, select _nbval_ on the left. There will
     be a message _This workflow has a workflow_dispatch event trigger._ and a _Run workflow_ button.
     Click on the button and select the branch that you want to test.
6. Test if the notebook works in [https://mybinder.org/](Binder) and if so, add _Launch Binder_ badges 
   to the README files.

Once your notebook passes in the CI and you have verified that everything looks good, make a Pull Request!

### Pull Requests (PRs)

1. If some time has passed since you made the fork, rebase or merge your fork to the
   openvino_notebooks main branch first.
2. Create your PR against the openvino_notebooks main branch.
3. Please create a description of what the notebook does with your PR. Screenshots are appreciated!
4. On making or updating a Pull Request, the tests in the CI will run again. Please keep an
   eye on them. If the tests fail and you think the issue is not related to your PR, please make a comment on your PR.

## Help

If you need help at any time, please open a
[discussion](https://github.com/openvinotoolkit/openvino_notebooks/issues)! If you think one of the
guidelines is too strict, or should not apply to you, feel free to ask about that too.
