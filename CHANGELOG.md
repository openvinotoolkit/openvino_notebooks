# Release 0.0.5 (2020-08-18T16:58:20)

- :truck: include thoth yaml file for auto updates
- :arrow_up: update base image to v0.15.0 and add version file (#8)
- :jack_o_lantern: include only necessary oc binaries (#6)
- Update jupyterlab/hub (#7)
- :truck: include aicoe-ci configuration file
- Create OWNERS
- Support python 3.6 as the base python for the image
- Install pipenv via pip
- :pushpin: Relock
- Relock requirements
- Hinterland is not enabled by default
- Move custom notebook config to /etc/jupyter
- Only update the relevant submodule
- Fixed removed source files in rsync
- Added LICENSE
- No need for the recursive pull
- Added a step to update submodules
- Use SSH key to checkout the repo
- Updated create-pull-request @v2
- Use custom PAT to clone thoth-station/jupyter-notebooks
- Added README.md
- Fixed custom .jupyter/ config files
- Added custom Jupyter config
- Install nbextensions contrib
- Updated assemble script to use /opt/app-root
- Added requirements files
- Updated builder scripts
- Imported original repository files

## Release 0.0.6 (2020-10-07T17:39:48)
### Features
* :rocket: update owners file
* Add support for cloning branch repo (#13)
* :wrench: patch for downloading oc client
* :truck: include aicoe-ci configuration file
### Improvements
* :books: updated README.md
* :books: updated README.md

## Release 0.0.7 (2021-01-28T05:03:21)
### Features
* :guardsman: clear github workflows from the s2i-minimal-notebook (#26)
* exclude cached files created from the build process (#28)
* upgrade the s2i-thoth base image to stable version
* :maple_leaf: Relock the Pipfile.lock
### Improvements
* :maple_leaf: updated requirement files (#25)
