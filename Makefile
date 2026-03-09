.PHONY: fmt lint lint-fix typechk

help:
	@echo "Available targets:"
	@echo "  fmt              - Format the code using Ruff"
	@echo "  lint             - Check linting of the code using Ruff"
	@echo "  lint-fix         - Check and fix linting if the code using Ruff"
	@echo "  typechk          - Type check the code using mypy"
	@echo "  help             - Show this help message"

fmt:
	uv run --dev ruff format

lint:
	uv run --dev ruff check

lint-fix:
	uv run --dev ruff check --fix

typechk:
	uv run --dev mypy .
