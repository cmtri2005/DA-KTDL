# Phase 7 Result Analysis

## 1. Phase 7 Run Summary

Phase 7 đã được chạy trên Google Colab với output tại:

```text
outputs/phase7_classification_outputs/
```

Training input chính:

```text
classify_quality_combined.jsonl
```

Các representation đã train trong run này:

```text
hybrid
quality_hybrid_top5
quality_hybrid_banded
```

Model:

```text
SciBERT: allenai/scibert_scivocab_uncased
```

Số mẫu:

```text
num_examples = 10000
num_train    = 8000
num_val      = 2000
best_epoch   = 3
```

## 2. Triple Quality Calibration

Phase 7 dùng threshold:

```text
high_threshold = 0.75
```

Summary:

| split | n_documents | n_triples | avg_triples_per_doc | avg_quality_score | avg_high_triples_per_doc | avg_low_triples_per_doc | zero_triple_documents |
|---|---:|---:|---:|---:|---:|---:|---:|
| cluster | 5000 | 33587 | 6.7174 | 0.769383 | 3.4678 | 3.2496 | 74 |
| classify | 10000 | 68084 | 6.8084 | 0.768498 | 3.4960 | 3.3124 | 145 |
| all | 15000 | 101671 | 6.7781 | 0.768790 | 3.4866 | 3.2915 | 219 |

Nhận xét:

- Threshold `0.75` là hợp lý hơn `0.65` và `0.85`.
- `0.65` quá lỏng: gần như mọi triple đều thành high-quality.
- `0.85` quá gắt: nhiều document gần như không còn high-quality triple.
- `0.75` tạo phân tách high/low khá cân bằng, nên `quality_hybrid_banded` có ý nghĩa ablation thật sự.

## 3. Classification Results

| representation | accuracy | macro-F1 | weighted-F1 | MCC | Cohen's kappa | top-3 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| hybrid | 0.8235 | 0.5688 | 0.8183 | 0.7813 | 0.7811 | 0.9710 |
| quality_hybrid_banded | 0.8210 | 0.5594 | 0.8153 | 0.7783 | 0.7781 | 0.9700 |
| quality_hybrid_top5 | 0.8190 | 0.5619 | 0.8141 | 0.7759 | 0.7757 | 0.9710 |

Delta so với baseline `hybrid`:

| representation | delta accuracy | delta macro-F1 | delta weighted-F1 | delta MCC |
|---|---:|---:|---:|---:|
| quality_hybrid_banded | -0.0025 | -0.0094 | -0.0030 | -0.0030 |
| quality_hybrid_top5 | -0.0045 | -0.0069 | -0.0042 | -0.0054 |

## 4. Main Conclusion

Trong run này, **quality filtering chưa cải thiện kết quả so với baseline `hybrid`**.

Baseline `hybrid = abstract [SEP] all triples` vẫn là cấu hình tốt nhất trong 3 representation đã chạy:

```text
hybrid accuracy  = 0.8235
hybrid macro-F1  = 0.5688
```

Hai quality variants đều thấp hơn:

```text
quality_hybrid_top5     macro-F1 giảm 0.0069
quality_hybrid_banded   macro-F1 giảm 0.0094
```

Điều này gợi ý rằng:

1. Raw triples không gây nhiễu đủ lớn để việc lọc rule-based giúp model tốt hơn.
2. Một số triple bị xem là "low quality" vẫn có thể chứa tín hiệu hữu ích cho classifier.
3. SciBERT có thể tự học cách bỏ qua phần triple nhiễu khi toàn bộ abstract vẫn còn trong input.
4. Rule-based quality score hiện tại còn thô, chưa phản ánh đầy đủ giá trị semantic của triple.

## 5. Per-Label Effects

### 5.1. `quality_hybrid_top5`

Một vài label được lợi:

| label | hybrid F1 | top5 F1 | delta |
|---|---:|---:|---:|
| hep-ex | 0.4286 | 0.4615 | +0.0330 |
| q-bio | 0.5000 | 0.5161 | +0.0161 |
| hep-ph | 0.7797 | 0.7863 | +0.0067 |
| physics | 0.6615 | 0.6641 | +0.0026 |

Một vài label bị giảm:

| label | hybrid F1 | top5 F1 | delta |
|---|---:|---:|---:|
| q-fin | 0.7826 | 0.7273 | -0.0553 |
| hep-th | 0.5833 | 0.5507 | -0.0326 |
| eess | 0.6269 | 0.6010 | -0.0259 |
| nucl-th | 0.6923 | 0.6667 | -0.0256 |

### 5.2. `quality_hybrid_banded`

Một vài label được lợi:

| label | hybrid F1 | banded F1 | delta |
|---|---:|---:|---:|
| hep-ex | 0.4286 | 0.5000 | +0.0714 |
| hep-ph | 0.7797 | 0.8103 | +0.0307 |
| gr-qc | 0.6947 | 0.7021 | +0.0074 |
| physics | 0.6615 | 0.6667 | +0.0051 |

Một vài label bị giảm:

| label | hybrid F1 | banded F1 | delta |
|---|---:|---:|---:|
| nlin | 0.2000 | 0.0000 | -0.2000 |
| eess | 0.6269 | 0.6000 | -0.0269 |
| nucl-th | 0.6923 | 0.6667 | -0.0256 |
| econ | 0.3333 | 0.3158 | -0.0175 |

Nhận xét:

- Quality filtering có giúp một số nhãn nhỏ như `hep-ex`, `hep-ph`.
- Tuy nhiên lợi ích này không đủ để bù cho các nhãn bị giảm như `nlin`, `q-fin`, `eess`, `hep-th`.
- Vì Macro-F1 lấy trung bình đều theo label, sự sụt giảm ở các label nhỏ có thể kéo điểm macro xuống rõ rệt.

## 6. Comparison With Phase 5

So với output Phase 5 trước đó, Phase 7 Colab run có baseline `hybrid` khá mạnh:

```text
Phase 5 best accuracy  ≈ 0.8140
Phase 5 best macro-F1  ≈ 0.5314

Phase 7 hybrid accuracy = 0.8235
Phase 7 hybrid macro-F1 = 0.5688
```

Tuy nhiên cần diễn giải cẩn thận:

- Phase 5 có thêm propagated cluster signal.
- Phase 7 Colab direct training không dùng propagation.
- Phase 7 Colab trainer dùng setup training đơn giản/fixed hyperparameters, không hoàn toàn giống Phase 5 pipeline Optuna.

Vì vậy không nên kết luận Phase 7 "thắng Phase 5" một cách trực tiếp. Kết luận an toàn hơn là:

> Trong cùng Phase 7 run, baseline `hybrid` mạnh hơn các quality-filtered variants.

## 7. What This Proves

Phase 7 trả lời được câu hỏi nghiên cứu ban đầu:

> Nếu triples có nhiễu, việc chấm điểm và chỉ dùng triples chất lượng cao có cải thiện Hybrid representation so với Hybrid dùng toàn bộ triples không?

Với kết quả hiện tại:

```text
Không. Rule-based triple quality filtering chưa cải thiện so với hybrid dùng toàn bộ triples.
```

Đây vẫn là kết quả có giá trị, vì nó chứng minh rằng:

- Chỉ lọc triples bằng heuristic đơn giản chưa đủ.
- Triple quality không chỉ phụ thuộc vào rule tag, phrase length, relation frequency, sentence length.
- Cần hướng mạnh hơn nếu muốn tận dụng triples tốt hơn, ví dụ:
  - triple extractor tốt hơn,
  - learned triple quality scorer,
  - graph-aware model,
  - attention/gating mechanism thay vì cắt bỏ triples.

## 8. Caveats

Run hiện tại mới gồm 3 representations:

```text
hybrid
quality_hybrid_top5
quality_hybrid_banded
```

Chưa chạy:

```text
abstract
triples
concatenate
quality_hybrid_top50
```

Vì vậy nếu muốn report Phase 7 đầy đủ theo plan ban đầu, nên chạy thêm ít nhất:

```text
quality_hybrid_top50
```

Nếu đủ thời gian GPU, chạy đủ bộ:

```text
abstract
triples
concatenate
hybrid
quality_hybrid_top5
quality_hybrid_top50
quality_hybrid_banded
```

Tuy nhiên, để trả lời câu hỏi cốt lõi "quality filtering có hơn hybrid không?", run hiện tại đã có bằng chứng khá rõ: **chưa hơn**.

