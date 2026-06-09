# Phase 7 Report - Triple-Quality-Aware Hybrid Classification

## Research Question

Phase 7 kiểm chứng giả thuyết: nếu triples có nhiễu, việc chấm điểm và chỉ dùng triples chất lượng cao có cải thiện classification so với `hybrid = abstract [SEP] all triples` hay không.

Phase 7 cố ý không dùng Phase 4 propagation trong experiment chính. Như vậy delta metric phản ánh tác động của triple-quality filtering, không bị lẫn với cluster signal.

## Triple Quality Summary

| split | n_documents | n_triples | avg_triples_per_doc | avg_quality_score | max_quality_score | avg_high_triples_per_doc | avg_low_triples_per_doc | zero_triple_documents | high_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cluster | 5000 | 33587 | 6.7174 | 0.769383 | 0.979671 | 6.6966 | 0.0208 | 74 | 0.65 |
| classify | 10000 | 68084 | 6.8084 | 0.768498 | 0.979671 | 6.7871 | 0.0213 | 145 | 0.65 |
| all | 15000 | 101671 | 6.778067 | 0.76879 | 0.979671 | 6.756933 | 0.021133 | 219 | 0.65 |

## Classification Ablation

### Results Sorted by Accuracy

| representation | model | accuracy | macro_f1 | weighted_f1 | mcc | top3_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| abstract | scibert | 0.8245 | 0.5771 | 0.8194 | 0.7826 | 0.9690 |
| hybrid | scibert | 0.8235 | 0.5688 | 0.8183 | 0.7813 | 0.9710 |
| concatenate | scibert | 0.8220 | 0.5613 | 0.8165 | 0.7795 | 0.9690 |
| quality_hybrid_banded | scibert | 0.8210 | 0.5594 | 0.8153 | 0.7783 | 0.9700 |
| quality_hybrid_top5 | scibert | 0.8190 | 0.5619 | 0.8141 | 0.7759 | 0.9710 |
| quality_hybrid_top50 | scibert | 0.8190 | 0.5565 | 0.8132 | 0.7758 | 0.9720 |
| triples | scibert | 0.7352 | 0.3754 | 0.7143 | 0.6671 | 0.8980 |

### Delta vs Hybrid Baseline

Hybrid baseline: accuracy=0.8235, macro-F1=0.5688.

| representation | model | delta_accuracy | delta_macro_f1 | delta_weighted_f1 |
| --- | --- | --- | --- | --- |
| abstract | scibert | 0.0010 | 0.0083 | 0.0012 |
| quality_hybrid_top5 | scibert | -0.0045 | -0.0069 | -0.0042 |
| concatenate | scibert | -0.0015 | -0.0074 | -0.0018 |
| quality_hybrid_banded | scibert | -0.0025 | -0.0094 | -0.0030 |
| quality_hybrid_top50 | scibert | -0.0045 | -0.0122 | -0.0051 |
| triples | scibert | -0.0883 | -0.1934 | -0.1040 |

### Interpretation Checklist

- Nếu quality variants có macro-F1 cao hơn `hybrid`, filtering giúp giảm triple noise.
- Nếu accuracy tăng nhưng macro-F1 không tăng, lợi ích chủ yếu nằm ở class lớn.
- Nếu quality variants thấp hơn `hybrid`, rule-based scoring hiện tại chưa đủ hoặc đã loại mất triple hữu ích.
- Nếu `triples` vẫn thấp hơn `abstract/hybrid`, triples-only vẫn thiếu ngữ cảnh so với abstract đầy đủ.

## Per-Label Effects

Các label hưởng lợi/hại nhất so với baseline `hybrid`:

### Largest Gains

| representation | model_alias | label | support | baseline_f1 | phase7_f1 | delta_f1_vs_hybrid |
| --- | --- | --- | --- | --- | --- | --- |
| triples | scibert | econ | 14 | 0.3333333333333333 | 0.43478260869565216 | 0.10144927536231885 |
| quality_hybrid_banded | scibert | hep-ex | 11 | 0.42857142857142855 | 0.5 | 0.07142857142857145 |
| quality_hybrid_top50 | scibert | hep-ex | 11 | 0.42857142857142855 | 0.5 | 0.07142857142857145 |
| abstract | scibert | math-ph | 17 | 0.17391304347826086 | 0.23076923076923078 | 0.05685618729096992 |
| abstract | scibert | hep-ex | 11 | 0.42857142857142855 | 0.4827586206896552 | 0.054187192118226646 |
| abstract | scibert | q-bio | 15 | 0.5 | 0.5517241379310345 | 0.051724137931034475 |
| abstract | scibert | hep-ph | 58 | 0.7796610169491526 | 0.8173913043478261 | 0.03773028739867357 |
| quality_hybrid_top50 | scibert | hep-ph | 58 | 0.7796610169491526 | 0.8141592920353983 | 0.0344982750862457 |
| quality_hybrid_top5 | scibert | hep-ex | 11 | 0.42857142857142855 | 0.46153846153846156 | 0.032967032967033016 |
| quality_hybrid_banded | scibert | hep-ph | 58 | 0.7796610169491526 | 0.8103448275862069 | 0.030683810637054276 |
| abstract | scibert | gr-qc | 40 | 0.6947368421052632 | 0.723404255319149 | 0.028667413213885773 |
| concatenate | scibert | stat | 60 | 0.4954128440366973 | 0.5185185185185185 | 0.023105674481821215 |
| abstract | scibert | econ | 13 | 0.3333333333333333 | 0.35294117647058826 | 0.019607843137254943 |
| quality_hybrid_top50 | scibert | stat | 60 | 0.4954128440366973 | 0.5142857142857142 | 0.01887287024901696 |
| abstract | scibert | stat | 60 | 0.4954128440366973 | 0.5137614678899083 | 0.01834862385321101 |

### Largest Drops

| representation | model_alias | label | support | baseline_f1 | phase7_f1 | delta_f1_vs_hybrid |
| --- | --- | --- | --- | --- | --- | --- |
| triples | scibert | nucl-th | 5 | 0.6923076923076923 | 0.0 | -0.6923076923076923 |
| triples | scibert | q-fin | 8 | 0.782608695652174 | 0.2 | -0.5826086956521739 |
| triples | scibert | hep-ex | 13 | 0.42857142857142855 | 0.0 | -0.42857142857142855 |
| triples | scibert | gr-qc | 25 | 0.6947368421052632 | 0.27906976744186046 | -0.41566707466340275 |
| triples | scibert | q-bio | 23 | 0.5 | 0.25 | -0.25 |
| triples | scibert | quant-ph | 84 | 0.8481012658227848 | 0.6171428571428571 | -0.23095840867992767 |
| triples | scibert | nlin | 6 | 0.2 | 0.0 | -0.2 |
| quality_hybrid_banded | scibert | nlin | 9 | 0.2 | 0.0 | -0.2 |
| triples | scibert | hep-ph | 44 | 0.7796610169491526 | 0.5853658536585366 | -0.19429516329061602 |
| triples | scibert | eess | 102 | 0.6268656716417911 | 0.4375 | -0.18936567164179108 |
| triples | scibert | math-ph | 17 | 0.17391304347826086 | 0.0 | -0.17391304347826086 |
| triples | scibert | physics | 155 | 0.6615384615384615 | 0.49491525423728816 | -0.16662320730117336 |
| triples | scibert | cond-mat | 153 | 0.8421052631578947 | 0.6848484848484848 | -0.15725677830940987 |
| quality_hybrid_top50 | scibert | econ | 13 | 0.3333333333333333 | 0.2222222222222222 | -0.1111111111111111 |
| triples | scibert | astro-ph | 138 | 0.9370629370629371 | 0.8363636363636363 | -0.10069930069930078 |

## Generated Files

- `phase7_report.md`
- `triple_quality_summary.csv`
- `results_table_phase7.csv`
- `per_label_delta_vs_hybrid.csv`
