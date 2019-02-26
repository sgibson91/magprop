# Makefile to run Monte Carlo optimisations and analyses on the Short Gamma-Ray
# Burst sample in the sgibson91/magprop repo


.PHONY: setup test test-cov clean


setup:
	mkdir -p plots


test:
	source activate kernel
	python -m pytest -vvv


test-cov:
	source activate kernel
	coverage run -m pytest -vvv
	coverage report
	coverage html
	@echo "Visit htmlcov/index.html in a browser to see interactive code coverage of the test suite"


clean:
	rm -f plots/*.png
	rm -rf plots
	rm -f htmlcov/*
	rm -rf htmlcov
