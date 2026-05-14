# Makefile for Triples and Knowledge-Infused Embeddings Project

.PHONY: help install setup phase1 phase2 phase3 phase4 phase5 phase6 smoke clean

help:
	@echo "Available targets:"
	@echo "  install  - Install dependencies"
	@echo "  setup    - Setup environment and data"
	@echo "  phase1   - Run Phase 1 (data preparation)"
	@echo "  phase2   - Run Phase 2 (embeddings)"
	@echo "  phase3   - Run Phase 3 (clustering)"
	@echo "  phase4   - Run Phase 4 (propagation)"
	@echo "  phase5   - Run Phase 5 (classification)"
	@echo "  phase6   - Run Phase 6 (reporting)"
	@echo "  smoke    - Run quick smoke test"
	@echo "  all      - Run all phases"
	@echo "  clean    - Remove output directories"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_sci_md

setup: install
	mkdir -p outputs/phase1_data outputs/phase2_embeddings outputs/phase3_clustering
	mkdir -p outputs/phase4_cluster_propagation outputs/phase5_classification
	mkdir -p reports/figures

phase1:
	python -m data_processing.pipeline --output-dir outputs/phase1_data

phase2:
	python -m embeddings --output-dir outputs/phase2_embeddings

phase3:
	python -m clustering --output-dir outputs/phase3_clustering

phase4:
	python -m propagation --output-dir outputs/phase4_cluster_propagation

phase5:
	python -m classification --output-dir outputs/phase5_classification

phase6:
	python phase6_reporting.py --output-dir reports/

all: phase1 phase2 phase3 phase4 phase5 phase6

smoke:
	@echo "Running smoke test on 100 documents..."
	python -m data_processing.pipeline --output-dir outputs/phase1_smoke --sample-size 100
	python -m embeddings --output-dir outputs/phase2_smoke --sample-size 100
	python -m clustering --output-dir outputs/phase3_smoke --sample-size 100

clean:
	rm -rf outputs/phase*_data outputs/phase*_embeddings outputs/phase*_clustering
	rm -rf outputs/phase*_propagation outputs/phase*_classification
	rm -rf reports/*.csv reports/*.md

.DEFAULT_GOAL := help
