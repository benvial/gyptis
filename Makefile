
SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: clean lint req doc help dev

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = gyptis
PYTHON_INTERPRETER = python3
HOSTING = gitlab
VERSION=$(shell python3 -c "import gyptis; print(gyptis.__version__)")
BRANCH=$(shell git branch --show-current)
URL=$(shell python3 -c "import gyptis; print(gyptis.__website__)")
LESSC=$(PROJECT_DIR)/docs/node_modules/less/bin/lessc
GITLAB_PROJECT_ID=22161961
GITLAB_GROUP_ID=11118791

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif


ifdef TEST_PARALLEL
TEST_ARGS=-n auto #--dist loadscope
endif


	

message = @make -s printmessage RULE=${1}



printmessage: 
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/^/---/" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} | grep "\---${RULE}---" \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=0 \
		-v col_on="$$(tput setaf 4)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s ", col_on, -indent, ">>>"; \
		n = split($$3, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i] ; \
		} \
		printf "%s ", col_off; \
		printf "\n"; \
	}' 

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Set up python interpreter environment
env:
ifeq (True,$(HAS_CONDA))
		@echo -e ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo -e ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo -e ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo -e ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test if python environment is setup correctly
testenv:
	$(call message,${@})
	source activate $(PROJECT_NAME); \
	$(PYTHON_INTERPRETER) dev/testenv.py

## Install Python dependencies
req:
	$(call message,${@})
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install Python dependencies for dev and test
dev:
	@$(PYTHON_INTERPRETER) -m pip install -r dev/requirements.txt
	
## Clean generated files
cleangen:
	$(call message,${@})
	@find . -not -path "./tests/data/*" | grep -E "(\.pvd|\.xdmf|\.msh|\.pvtu|\.vtu|\.pvd|jitfailure*|tmp|__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf .pytest_cache $(PROJECT_NAME).egg-info/ build/ dist/ tmp/ htmlcov/
	
## Clean documentation
cleandoc:
	$(call message,${@})
	@cd docs && make -s clean

## Clean project
clean: cleantest cleangen cleanreport
	$(call message,${@})

## Clean all project
clean-all: cleandoc clean
	$(call message,${@})

## Lint using flake8
lint:
	$(call message,${@})
	@flake8 --exit-zero --ignore=E501 setup.py $(PROJECT_NAME)/ tests/*.py examples/

## Check for duplicated code
dup:
	$(call message,${@})
	@pylint --exit-zero -f colorized --disable=all --enable=similarities $(PROJECT_NAME)


## Clean code stats
cleanreport:
	$(call message,${@})
	@rm -f pylint.html

## Report code stats
report: cleanreport
	$(call message,${@})
	@pylint $(PROJECT_NAME) | pylint-json2html -f jsonextended -o pylint.html


## Check for missing docstring
dcstr:
	$(call message,${@})
	@pydocstyle ./$(PROJECT_NAME)  || true

## Metric for complexity
rad:
	$(call message,${@})
	@radon cc ./$(PROJECT_NAME) -a -nb

## Run all code checks
lint-all: lint dup dcstr rad
	$(call message,${@})

## Reformat code
style:
	$(call message,${@})
	@isort .
	@black .

## Push to gitlab
gl:
	$(call message,${@})
	@git add -A
	@read -p "Enter commit message: " MSG; \
	git commit -a -m "$$MSG"
	@git push origin $(BRANCH)
	

## Show gitlab repository
repo:
	xdg-open https://gitlab.com/gyptis/gyptis


## Clean, reformat and push to gitlab
save: clean style gl
	$(call message,${@})
	

## Push to gitlab (skipping continuous integration)
gl-noci:
	$(call message,${@})
	@git add -A
	@read -p "Enter commit message: " MSG; \
	git commit -a -m "$$MSG [skip ci]"
	@git push origin $(BRANCH)
	
	

## Clean, reformat and push to gitlab (skipping continuous integration)
save-noci: clean style gl-noci
	$(call message,${@})
	


## Make documentation css
less:
	$(call message,${@})
	@rm -f docs/_custom/static/css/*.css
	@cd docs/_custom/static/css/less && \
	$(LESSC) custom_styles.less  ../custom_styles.css && \
	$(LESSC) custom_gallery.less  ../custom_gallery.css && \
	$(LESSC) custom_pygments.less  ../custom_pygments.css && \
	$(LESSC) custom_notebook.less  ../custom_notebook.css

	
## Install requirements for building documentation
doc-req:
	$(call message,${@})
	@cd docs && pip install -r requirements.txt && npm install lessc


## Build html documentation (only updated examples)
doc: less
	$(call message,${@})
	@cd docs && make -s html && make -s postpro-html


## Build html documentation (without examples)
doc-noplot: less
	$(call message,${@})
	@cd docs && make -s clean && make -s html-noplot && make -s postpro-html

## Show locally built html documentation in a browser
show-doc:
	$(call message,${@})
	@cd docs && make -s show
	
## Show locally built pdf documentation
show-pdf:
	$(call message,${@})
	@cd docs && make -s show-pdf
	
## Build pdf documentation (only updated examples)
pdf:
	$(call message,${@})
	@cd docs && make -s latexpdf
	
## Build pdf documentation (without examples)
pdf-noplot:
	$(call message,${@})
	@cd docs && make -s latexpdf-noplot

## Clean test coverage reports
cleantest:
	$(call message,${@})
	@rm -rf .coverage* htmlcov coverage.xml

## Run the test suite
test: cleantest
	$(call message,${@})
	@export MPLBACKEND=agg && unset GYPTIS_ADJOINT && pytest ./tests \
	--cov=./$(PROJECT_NAME) --cov-report term --durations=0 $(TEST_ARGS) 
	@export MPLBACKEND=agg && GYPTIS_ADJOINT=1 pytest ./tests \
	--cov=./$(PROJECT_NAME) --cov-append --cov-report term \
	--cov-report html --cov-report xml --durations=0 $(TEST_ARGS)

## Run the test suite (parallel)
testpara: cleantest
	$(call message,${@})
	@make -s test TEST_PARALLEL=1
	

## Copy the coverage html into documentation
covdoc:
	$(call message,${@})
	@ls docs/_build/html/ || make doc
	@ls htmlcov/ || make -s test && mv htmlcov/ docs/_build/html/coverage/
	

## Tag and push tags
tag: clean style
	$(call message,${@})
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Version v$(VERSION)"
	@git add -A
	git commit -a -m "Publih v$(VERSION)"
	@git push origin $(BRANCH)
	@git tag v$(VERSION) && git push --tags || echo Ignoring tag since it already exists
	
## Create a release
release:
	$(call message,${@})
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@gitlab project-release create --project-id $(GITLAB_PROJECT_ID) \
	--name "version $(VERSION)" --tag-name "v$(VERSION)" --description "Released version $(VERSION)"
                                     

## Package 
package: setup.py
	$(call message,${@})
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@rm -f dist/*
	python3 setup.py sdist
	python3 setup.py bdist_wheel --universal

## Upload to pypi
pypi: package
	$(call message,${@})
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@twine upload dist/*

## Make checksum for release
checksum:
	@echo v$(VERSION)
	$(eval SHA256 := $(shell curl -sL https://gitlab.com/gyptis/gyptis/-/archive/v$(VERSION)/gyptis-v$(VERSION).tar.gz | openssl sha256 | cut  -c10-))
	@echo $(SHA256)
	# @sed -i "s/sha256: .*/sha256: $(SHA256)/" ../test/meta.yaml
	# @sed -i "s/number: .*/number: 0/" ../test/meta.yaml
	# @sed -i "s/{% set version = .*/{% set version = \"$(VERSION)\" %}/" ../test/meta.yaml
	# 
# 
# ## Update conda	
# conda: checksum
# 	@echo checksum is $(SHA256)
# 	## 

## Update conda	
conda: checksum
	@cd ../gyptis-feedstock && \
# create branch in forked repo
	git branch v$(VERSION) && \
# change recipe: reset build number, update version and checksum
	@sed -i "s/sha256: .*/sha256: $(SHA256)/" recipe/test/meta.yaml && \
	@sed -i "s/number: .*/number: 0/" recipe/test/meta.yaml && \
	@sed -i "s/{% set version = .*/{% set version = \"$(VERSION)\" %}/" recipe/test/meta.yaml && \
# update
	@git add -A && \
	git commit -a -m "New version $(VERSION)" && \
	@git push origin $(BRANCH) && \
# create pull request
	git pull-request --no-edit --browse



## Publish release on pypi and conda-forge
publish: tag release pypi conda

	
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################


# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>

help:
	@echo -e "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo -e
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
