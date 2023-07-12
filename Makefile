help:
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

clean: ## Deleted unwanted files and folders
	echo "Cleaning make_tabular_data"
	rm -rf build dist htmlcov *.egg-info
	find . -name "*.png" -type f -delete
	find . -name "*.coverage" -type f -delete
	find . -name "*.csv" -type f -delete
	find . -name "*.pkl" -type f -delete
	find . -name "*.h5" -type f -delete
	find . -name "output" -type d -delete


lint_local: ## Lints Python Directories Locally
	flake8 --statistics --count --exit-zero

test_local: ## Runs unit tests with coverage and opens results in the browser
	python3 -m pytest tests -s -v --cov synthetic_data --cov-report term-missing

examples_list: ## List all available examples
	ls examples | egrep -v .py

pre_commit: ## Run pre-commit
	pre-commit install
	pre-commit run
