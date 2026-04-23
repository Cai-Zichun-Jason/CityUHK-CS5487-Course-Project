PYTHON := python3

.PHONY: all build checkenv train base eval challenge clean help

help:
	@echo "Usage:"
	@echo "  make all        - run full pipeline: build -> checkenv -> train -> base -> challenge -> eval"
	@echo "  make build      - install dependencies (requires Python 3.8+)"
	@echo "  make checkenv   - verify Python environment and data files"
	@echo "  make train      - train all models, save to data/trained-model/"
	@echo "  make base       - evaluate all models on the base test set"
	@echo "  make eval       - run hyperparameter sensitivity analysis scripts"
	@echo "  make challenge  - evaluate all models on the challenge test set"
	@echo "  make clean      - remove trained models and __pycache__"

all: build checkenv train base challenge eval

build:
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "Error: $(PYTHON) not found. Please install Python 3.8+."; exit 1; }
	$(PYTHON) -m pip install -r requirements.txt || $(PYTHON) -m pip install --break-system-packages -r requirements.txt

checkenv:
	@$(PYTHON) --version
	$(PYTHON) check_env.py

train:
	$(PYTHON) main.py --mode train --model all

base:
	$(PYTHON) main.py --mode base --model all

eval:
	$(PYTHON) main.py --mode eval

challenge:
	$(PYTHON) main.py --mode challenge --model all

clean:
	rm -rf data/trained-model/*
	rm -rf results/images/*
	rm -rf results/eval_figures/*
	find . -type d -name __pycache__ -exec rm -rf {} +
