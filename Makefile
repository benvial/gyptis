#
# ifeq ($(TRAVIS_OS_NAME),windows)
# 	SHELL := cmd
# else
# 	SHELL := /bin/bash
# endif

SHELL := /bin/bash

VERSION=$(shell python3 -c "import gyptis; print(gyptis.__version__)")


default:
	@echo "\"make save\"?"

tag:
	# Make sure we're on the master branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags

pipy: package
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	twine upload dist/*

package: setup.py
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	rm -f dist/*
	python3 setup.py sdist
	python3 setup.py bdist_wheel --universal


gh:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Pushing to github..."
	git add -A
	@read -p "Enter commit message: " MSG; \
	git commit -a -m "$$MSG"
	git push

publish: tag pipy

test:
	pytest -v ./tests -s --cov=./ --cov-report html


clean:
	@find . | grep -E "(jitfailure*|__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf .pytest_cache gyptis.egg-info/ build/ dist/ tmp/
	cd docs && make clean


lstmp:
	@find . -type d -name 'tmp*'


rmtmp:
	@find . -type d -name 'tmp*' | xargs rm -rf


lint:
	flake8 setup.py gyptis/ tests/*.py

style:
	@echo "Styling..."
	black setup.py gyptis/ tests/*.py

less:
	cd docs && make less

webdoc: less
	cd docs && make clean && make html

webdoc-noplot: less
	cd docs && make clean && make html-noplot

latexpdf:
	cd docs && make latexpdf


latexpdf-noplots:
	cd docs && make latexpdf-noplots
	cp docs/_build/latex/gyptis.pdf docs/_build/html/_downloads/gyptis.pdf

deploydoc: clean webdoc latexpdf
	git add -A
	git commit -a -m "update docs"
	git checkout gh-pages
	git merge master
	git push origin gh-pages
	git checkout master


## Show html doc in browser
showdoc:
	$(BROWSER) ./docs/_build/html/index.html

save: clean style gh
