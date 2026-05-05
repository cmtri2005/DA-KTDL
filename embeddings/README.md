# Phase 2 - Embedding Generation

## Cách chạy

### 1. Cài dependencies

```bash
pip install -r requirements-phase2.txt
```



### 2. Chạy toàn bộ Phase 2

```bash
python -m phase2_embeddings \
  --phase1_output output_triples \
  --output_root outputs/phase2_embeddings \
  --splits all \
  --representations all \
  --models all \
  --batch_size 32 \
  --max_length 512 \
  --pooling mean \
  --device auto \
  --overwrite
```

### 3. Chạy một phần

Ví dụ chỉ chạy `cluster`, chỉ dùng `abstract` và `triples`, với 2 model:

```bash
python -m phase2_embeddings \
  --phase1_output output_triples \
  --output_root outputs/phase2_embeddings \
  --splits cluster \
  --representations abstract triples \
  --models minilm mpnet \
  --batch_size 64 \
  --max_length 512 \
  --pooling mean \
  --device cuda \
  --overwrite
```

## Ý nghĩa tham số

- `--phase1_output`
  - thư mục chứa output của Phase 1
- `--output_root`
  - thư mục gốc để lưu output của Phase 2
- `--splits`
  - chọn `cluster`, `classify` hoặc `all`
- `--representations`
  - chọn `abstract`, `triples`, `concatenate`, `hybrid` hoặc `all`
- `--models`
  - chọn `minilm`, `mpnet`, `specter`, `scibert` hoặc `all`
- `--batch_size`
  - số văn bản encode trong một batch
- `--max_length`
  - số token tối đa khi tokenize cho model transformer
- `--pooling`
  - cách gom token embedding thành document embedding cho `SciBERT` và `SPECTER`
  - nhận `mean` hoặc `cls`
- `--device`
  - thiết bị chạy: `auto`, `cpu`, `cuda`, `cuda:0`, ...
- `--overwrite`
  - ghi đè output đã tồn tại
- `--normalize`
  - bật L2-normalize cho embeddings
- `--no_normalize`
  - tắt L2-normalize

## Output

Mỗi job sẽ lưu:

- `embeddings.npy`
- `metadata.jsonl`
- `verification.json`
- `run_config.json`
