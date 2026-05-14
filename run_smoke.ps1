# Smoke test script - Quick validation on small dataset
# Usage: .\run_smoke.ps1

Write-Host "=== Phase 1: Data Preparation (Smoke Test) ===" -ForegroundColor Green
python -m data_processing.pipeline --output-dir outputs/phase1_smoke --sample-size 100

Write-Host "=== Phase 2: Embedding Generation (Smoke Test) ===" -ForegroundColor Green
python -m embeddings --output-dir outputs/phase2_smoke --sample-size 100

Write-Host "=== Phase 3: Clustering (Smoke Test) ===" -ForegroundColor Green
python -m clustering --output-dir outputs/phase3_smoke --sample-size 100

Write-Host "=== Smoke Test Complete ===" -ForegroundColor Green
Write-Host "Check outputs/phase*_smoke/ for results"
