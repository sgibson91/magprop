# Makefile to run Monte Carlo optimisations and analyses on the Short Gamma-Ray
# Burst sample in the sgibson91/magprop repo


SHELL=/bin/bash
CONDAROOT=/usr/local/anaconda3


.PHONY: setup test test-cov clean


setup:
	conda env create --force --file environment.yml
	mkdir -p plots


test:
	source $(CONDAROOT)/bin/activate && conda activate magnetar-env && which python && pwd


test-cov:
	source activate kernel
	coverage run -m pytest -vvv
	coverage report
	coverage html
	@echo "Visit htmlcov/index.html in a browser to see interactive code coverage of the test suite"


clean:
	@conda env remove -n magnetar-env
	@rm -f plots/*.png
	@rm -rf plots
	@rm -f htmlcov/*
	@rm -rf htmlcov
