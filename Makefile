VIRTUALENV_DIR=.venv
ACTIVATE=$(VIRTUALENV_DIR)/bin/activate

venv:
	@echo Creating venv for notebooks
	@python3 -m venv $(VIRTUALENV_DIR)

cache_openvino_packages:
	@echo Cache openvino packages
	@. $(ACTIVATE); python -m pip install --upgrade pip
	mkdir pipcache
	@. $(ACTIVATE); python -m pip install --cache-dir pipcache --no-deps openvino openvino-dev nncf
	cp -r pipcache pipcache_openvino
	@. $(ACTIVATE); python -m pip uninstall -y openvino openvino-dev nncf

install_dependencies:
	@echo Installing dependencies
	@. $(ACTIVATE); python -m pip install --upgrade pip
	@. $(ACTIVATE); python -m pip install -r .ci/dev-requirements.txt --cache-dir pipcache
	@. $(ACTIVATE); python -m ipykernel install --user --name openvino_env
	@. $(ACTIVATE); python -m pip freeze

check_install:
	@echo Checking installation
	@. $(ACTIVATE); python check_install.py

convert_notebooks: venv cache_openvino_packages install_dependencies check_install
	@echo Running notebooks
	@. $(ACTIVATE); bash .ci/convert_notebooks.sh
