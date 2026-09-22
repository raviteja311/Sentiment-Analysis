PYTHON ?= python
IMAGE ?= sentiment-analysis-api:local
PORT ?= 8000

.PHONY: help install install-dev install-train api ui test lint format \
        train-lr train-lstm train-gru train-bert evaluate \
        docker-build docker-run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

# --- environment -----------------------------------------------------------

install:  ## Install the CPU inference stack
	$(PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
		-r requirements/inference-cpu.txt

install-dev:  ## Install the inference stack plus test and lint tooling
	$(PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
		-r requirements/dev.txt

install-train:  ## Install everything needed to retrain
	$(PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
		-r requirements/train.txt

# --- run -------------------------------------------------------------------

api:  ## Serve the API on $(PORT)
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT)

ui:  ## Serve the Streamlit app
	$(PYTHON) -m streamlit run app/streamlit_app.py

# --- quality ---------------------------------------------------------------

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Check formatting and lint rules
	$(PYTHON) -m ruff check src api tests app
	$(PYTHON) -m black --check src api tests app

format:  ## Apply formatting and safe lint fixes
	$(PYTHON) -m ruff check --fix src api tests app
	$(PYTHON) -m black src api tests app

# --- weights ---------------------------------------------------------------

fetch-weights:  ## Download the large weights from the Hugging Face Hub
	$(PYTHON) -m src.artifacts

publish-weights:  ## Upload the large weights (requires `huggingface-cli login`)
	$(PYTHON) scripts/publish_weights.py

# --- training and evaluation ----------------------------------------------

train-lr:  ## Train the TF-IDF + LogisticRegression baseline (minutes, CPU)
	$(PYTHON) -m src.training.train_lr

train-lstm:  ## Train the Bi-LSTM (tens of minutes, CPU)
	$(PYTHON) -m src.training.train_lstm

train-gru:  ## Train the Bi-GRU (tens of minutes, CPU)
	$(PYTHON) -m src.training.train_gru

train-bert:  ## Fine-tune the Twitter-RoBERTa checkpoint (GPU strongly advised)
	$(PYTHON) -m src.training.train_bert

evaluate:  ## Score every available model and print the README table
	$(PYTHON) -m src.evaluate

# --- containers ------------------------------------------------------------

docker-build:  ## Build the CPU serving image
	docker build --target cpu -t $(IMAGE) .

docker-run:  ## Run the serving image on $(PORT)
	docker run --rm -p $(PORT):8000 $(IMAGE)

# --- housekeeping ----------------------------------------------------------

clean:  ## Remove caches and compiled files
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
