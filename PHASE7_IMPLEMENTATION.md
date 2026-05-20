# Plan Implement Phase 7 - Triple-Quality-Aware Hybrid Classification

## Summary

Triển khai Phase 7 như một extension tách riêng, không sửa lại Phase 1-6 core quá nhiều. Mục tiêu report-ready: tạo quality score cho từng triple, sinh các representation mới dựa trên triple chất lượng cao, chạy classification ablation cùng metrics Phase 5, rồi xuất bảng/markdown report để chứng minh filtering triples có giúp hay không.

Phase 7 sẽ tập trung trả lời câu hỏi nghiên cứu:

> Nếu triples có nhiễu, việc chấm điểm và chỉ dùng triples chất lượng cao có cải thiện Hybrid representation so với Hybrid dùng toàn bộ triples không?

## Key Implementation Changes

### 1. Tạo module Phase 7 mới

Thêm package mới `phase7_quality/` với CLI chạy được bằng:

```bash
python -m phase7_quality build
python -m phase7_quality classify
python -m phase7_quality report
```

Public CLI chính:

```bash
python -m phase7_quality build \
  --phase1_root outputs/phase1_data \
  --output_root outputs/phase7_quality \
  --splits cluster classify \
  --top_k 5 \
  --top_fraction 0.5 \
  --high_threshold 0.65
```

```bash
python -m phase7_quality classify \
  --quality_root outputs/phase7_quality \
  --output_root outputs/phase7_classification \
  --representations abstract triples concatenate hybrid quality_hybrid_top5 quality_hybrid_top50 quality_hybrid_banded \
  --models scibert \
  --max_length 128 \
  --train_size 0.8 \
  --optuna_trials 3 \
  --seed 42
```

```bash
python -m phase7_quality report \
  --classification_root outputs/phase7_classification \
  --quality_root outputs/phase7_quality \
  --output_dir reports/phase7
```

### 2. Triple quality scoring

Input lấy từ `outputs/phase1_data/{split}_combined.jsonl`, tận dụng sẵn:

```text
triples[].subj
triples[].rel
triples[].obj
triples[].source_sentence
triples[].rule_tag
```

Mỗi triple được chấm 4 feature, chuẩn hóa `[0, 1]`:

```text
dependency_rule_score
phrase_length_score
relation_frequency_score
source_sentence_score
```

Final score:

```text
quality_score =
  0.35 * dependency_rule_score
+ 0.25 * phrase_length_score
+ 0.25 * relation_frequency_score
+ 0.15 * source_sentence_score
```

Quy tắc cụ thể:

- `dependency_rule_score`
  - `dobj`: `1.00`
  - `attr`: `0.90`
  - `prep->pobj`: `0.65`
  - `prep->pcomp`: `0.60`
  - unknown/default: `0.55`

- `phrase_length_score`
  - Tính token count cho subject và object, lấy trung bình hai score.
  - 2-8 tokens: `1.00`
  - 1 token: `0.55`
  - 9-12 tokens: `0.80`
  - 13-20 tokens: `0.45`
  - >20 tokens: `0.25`
  - Nếu subject/object là generic pronoun như `we`, `this`, `that`, `which`, cap phrase score ở `0.70`.

- `relation_frequency_score`
  - Tính frequency của `rel` trên toàn corpus Phase 1 đã load, không dùng label.
  - `freq_norm = log1p(freq) / log1p(max_freq)`
  - `score = clip(1 - freq_norm, 0.25, 1.0)`
  - Nếu relation thuộc generic set `{be, have, use, do, make, show, present, study, consider}`, cap score ở `0.45`.

- `source_sentence_score`
  - <=40 tokens: `1.00`
  - 41-70 tokens: `0.75`
  - 71-100 tokens: `0.50`
  - >100 tokens: `0.25`
  - Nếu sentence có nhiều ký hiệu công thức/ký tự nhiễu, trừ thêm `0.15`, floor `0.10`.

### 3. Sinh representation mới

Với mỗi document, sort triples theo `quality_score` giảm dần, rồi tạo:

```text
quality_triples_top5
quality_hybrid_top5
quality_triples_top50
quality_hybrid_top50
quality_hybrid_banded
```

Định nghĩa:

```text
quality_triples_top5   = linearized top 5 triples
quality_hybrid_top5    = abstract [SEP] quality_triples_top5

quality_triples_top50  = top ceil(n_triples * 0.5)
quality_hybrid_top50   = abstract [SEP] quality_triples_top50

quality_hybrid_banded  = abstract [SEP] high confidence triples [SEP] low confidence triples
```

Với `quality_hybrid_banded`:

```text
high confidence triples = triples có score >= 0.65
low confidence triples  = triples có score < 0.65
```

Nếu document không có triple hoặc không giữ triple nào, quality hybrid text sẽ fallback thành `abstract`, không thêm `[SEP]` rỗng.

Output Phase 7 build:

```text
outputs/phase7_quality/
  quality_config.json
  relation_frequency.json
  cluster_quality_combined.jsonl
  classify_quality_combined.jsonl
  cluster_quality_hybrid_top5.jsonl/.csv
  classify_quality_hybrid_top5.jsonl/.csv
  ...
  triple_quality_rows.jsonl
  triple_quality_summary.csv
```

Mỗi record quality combined giữ thêm:

```text
avg_quality_score
max_quality_score
n_quality_high
n_quality_low
n_quality_top5
n_quality_top50
quality_triples
fmt_quality_hybrid_top5
fmt_quality_hybrid_top50
fmt_quality_hybrid_banded
```

### 4. Classification ablation cho Phase 7

Không dùng Phase 4 propagation trong experiment chính của Phase 7 để cô lập tác động của triple quality filtering.

Phase 7 classification sẽ reuse `classification.training.run_experiment()` nhưng tự build examples trực tiếp từ quality representations:

```text
model_text = selected representation text
clustering_representation = "none"
propagated_cluster_id = -1
propagation_confidence = 0.0
```

Default report-ready experiment:

```text
models: scibert
representations:
  abstract
  triples
  concatenate
  hybrid
  quality_hybrid_top5
  quality_hybrid_top50
  quality_hybrid_banded
```

Metrics lưu:

```text
accuracy
macro-F1
weighted-F1
top-3 accuracy
MCC
Cohen's kappa
macro ROC-AUC nếu tính được
per-label precision/recall/F1
```

Output:

```text
outputs/phase7_classification/
  results_table_phase7.csv
  results_table_phase7_best.csv
  per_label_metrics.csv
  per_label_delta_vs_hybrid.csv
  planned_experiments.json
  {representation}/{model}/experiment_summary.json
```

### 5. Phase 7 reporting

`python -m phase7_quality report` tạo:

```text
reports/phase7/phase7_report.md
reports/phase7/results_table_phase7.csv
reports/phase7/per_label_delta_vs_hybrid.csv
reports/phase7/triple_quality_summary.csv
reports/phase7/figures/quality_score_distribution.png
reports/phase7/figures/macro_f1_comparison.png
reports/phase7/figures/per_label_f1_delta.png
```

Report phải trả lời rõ:

- Quality filtering có tăng macro-F1 so với `hybrid` không?
- `top5`, `top50`, hay `banded` tốt hơn?
- Triple-only vẫn thấp hơn abstract/hybrid không?
- Label nào được lợi/hại nhất?
- Nếu không cải thiện, kết luận là rule-based triple quality chưa đủ, cần extractor tốt hơn hoặc graph-aware model.

## Test Plan

### Unit tests

Dùng `python -m unittest` cho scoring functions:

- `dependency_rule_score` trả đúng score cho `dobj`, `attr`, `prep->pobj`, unknown.
- `phrase_length_score` giảm khi subject/object quá ngắn hoặc quá dài.
- `relation_frequency_score` giảm với relation quá phổ biến và generic verbs.
- `source_sentence_score` giảm với sentence quá dài/nhiễu.
- `quality_score` luôn nằm trong `[0, 1]`.

### Data build smoke test

Chạy:

```bash
python -m phase7_quality build \
  --phase1_root outputs/phase1_data \
  --output_root outputs/phase7_quality_smoke \
  --splits classify \
  --limit 200 \
  --top_k 5 \
  --top_fraction 0.5
```

Acceptance:

- Sinh được `classify_quality_combined.jsonl`.
- Số dòng = 200.
- Không record nào mất `id`, `label`, `abstract`.
- `quality_score` không NaN/Inf.
- `quality_hybrid_top5` không giữ quá 5 triples/document.
- Document không có triples fallback về abstract.

### Classification smoke test

Chạy subset nhỏ:

```bash
python -m phase7_quality classify \
  --quality_root outputs/phase7_quality_smoke \
  --output_root outputs/phase7_classification_smoke \
  --representations hybrid quality_hybrid_top5 \
  --models minilm \
  --optuna_trials 1 \
  --epochs_min 1 \
  --epochs_max 1 \
  --max_length 128 \
  --seed 42
```

Acceptance:

- Có `results_table_phase7.csv`.
- Có đủ 2 rows cho 2 representations.
- Có `per_label_metrics.csv`.
- Training không crash khi representation có document zero triples.

### Full report acceptance

Sau full run:

- `reports/phase7/phase7_report.md` tồn tại.
- Bảng có baseline `abstract`, `triples`, `concatenate`, `hybrid`.
- Bảng có ít nhất 3 quality variants: `quality_hybrid_top5`, `quality_hybrid_top50`, `quality_hybrid_banded`.
- Report có delta macro-F1 của từng quality variant so với `hybrid`.
- Report nêu rõ nếu improvement không đạt kỳ vọng.

## Assumptions and Defaults

- Scope chọn theo mức **Report-ready**: implement được data, smoke, full experiment design, report; không bắt buộc chạy full heavy nhiều model ngay.
- Model full mặc định là `SciBERT` vì Phase 7 hypothesis nói SciBERT có khả năng hưởng lợi nhất từ structured scientific text.
- Không dùng propagation trong Phase 7 experiment chính để tránh lẫn tác động giữa cluster signal và triple-quality filtering.
- Relation frequency là thống kê unsupervised trên text/triples, không dùng label thật nên không xem là label leakage.
- Baseline so sánh chính là `hybrid` hiện có từ Phase 1.
- Nếu cần kết quả mạnh hơn sau report-ready run, mở rộng thêm `specter` và `minilm`, hoặc tăng `optuna_trials` lên 10.
