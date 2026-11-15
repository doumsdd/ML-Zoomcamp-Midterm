# Pokemon Type Predictor - Makefile
# Usage: make [target]
# Targets: setup, train, test, run, predict, docker-build, docker-run, lint

.PHONY: setup train test run predict docker-build docker-run clean help

help: ## Show help
	@echo "Available targets:"
	@echo "  setup       Install dependencies"
	@echo "  train       Train the model"
	@echo "  test        Run unit tests"
	@echo "  run         Run Flask API locally"
	@echo "  predict     Test prediction via curl"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run  Run Docker container"
	@echo "  clean       Remove generated files"

setup: ## Install dependencies
	pip install -r requirements.txt

train: ## Train the model
	python train.py

test: ## Run unit tests
	python -m pytest tests/ -v

run: ## Run Flask API on port 9696
	Python -m flask --app predict run --host 0.0.0.0 --port 9696

predict: ## Test prediction
	curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"name\":\"bulbasaur\"}"

docker-build: ## Build Docker image
	docker build -t pokemon-predictor .

docker-run: ## Run Docker container
	docker run -p 9696:9696 pokemon-predictor

clean: ## Remove generated files
	rm -f model.pkl feature_names.json training_metrics.json *.png
