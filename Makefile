# 作者: 晨星
# MetricForge Makefile —— 一键复现：环境 / 测试 / lint / demo / 发布
PY ?= python
VENV ?= .venv
INSTALL := $(PY) -m pip install --quiet

.PHONY: venv install install-dev test lint demo clean check

venv:
	$(PY) -m venv $(VENV)

install:
	$(INSTALL) -r requirements.txt

install-dev: install
	$(INSTALL) -r requirements-optional.txt
	$(INSTALL) pytest ruff

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check metricforge
	$(PY) -m ruff format --check metricforge

demo:
	$(PY) examples/run_demo.py

# 端到端校验：单测 + lint + demo
check: test lint demo

clean:
	rm -rf .pytest_cache benchmark.json .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
