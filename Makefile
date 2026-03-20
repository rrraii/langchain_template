.PHONY: test pytest lint format cov build clean doctor precommit

test:
	python -m unittest discover -s tests -p "test_*.py" -v

pytest:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

cov:
	pytest --cov=lc_templates --cov-report=term-missing

doctor:
	python -m lc_templates doctor

precommit:
	pre-commit run --all-files

build:
	python -m build

clean:
	if exist build rmdir /s /q build
	if exist dist rmdir /s /q dist
	if exist .pytest_cache rmdir /s /q .pytest_cache
