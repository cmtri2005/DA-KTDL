# To-do list tái hiện paper Triples and Knowledge-Infused Embeddings

Mục tiêu: tái hiện pipeline trong paper "Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents", sau đó mở rộng bằng một hướng NCKH khả thi dựa trên triples/knowledge graph.

## Phase 0 - Reproduction contract

- [ ] Chốt dataset: dùng Kaggle/Cornell arXiv metadata snapshot `dataset/arxiv-metadata-oai-snapshot.json`.
- [ ] Ghi lại ngày snapshot, số dòng, dung lượng, checksum nếu có thể.
- [ ] Cố định seed `42` cho shuffle, split, clustering, train/validation và model initialization.
- [ ] Tạo cấu trúc output chuẩn:
  - [ ] `outputs/phase1_data/`
  - [ ] `outputs/phase2_embeddings/`
  - [ ] `outputs/phase3_clustering/`
  - [ ] `outputs/phase4_classification/`
  - [ ] `outputs/phase5_extension/`
- [ ] Ghi mọi config chạy thí nghiệm vào JSON/YAML: dataset filter, representation mode, model, hyperparameters, metrics.
- [ ] Chấp nhận sai khác với paper ở điểm paper chỉ nói "recent years" nhưng không công bố chính xác mốc năm lọc. Mọi mốc năm dùng trong repo phải được ghi rõ.

## Phase 1 - Data preparation, triples, knowledge graph edges

- [x] Load arXiv metadata JSONL.
- [x] Lọc record có `abstract`, `categories`, `update_date`.
- [x] Map label từ category đầu tiên, ví dụ `cs.AI -> cs`.
- [x] Clean abstract bằng lowercase và whitespace normalization.
- [x] Chia non-overlap:
  - [x] 5.000 documents cho clustering.
  - [x] 10.000 documents cho classification.
- [x] Extract triples từ abstract bằng spaCy/scispaCy:
  - [x] verb/AUX làm relational anchor.
  - [x] subject từ `nsubj`, `nsubjpass`.
  - [x] object từ `dobj`, `attr`.
  - [x] fallback object từ `prep -> pobj/pcomp`.
  - [x] lưu `source_sentence` để truy vết provenance.
- [x] Linearize triple `(s, r, o)` thành câu ngắn: `Subject relation object.`
- [x] Lưu document-level KG edge list `source, relation, target, source_sentence, rule_tag`.
- [x] Tạo 4 representation đúng Section 3.3:
  - [x] `Abstract`: cleaned abstract.
  - [x] `Triples`: chỉ linearized triples.
  - [x] `Abstract+Triples`: nối phẳng abstract và triples.
  - [x] `Hybrid`: `abstract [SEP] triples`.
- [x] Xuất JSONL/CSV cho từng split và từng representation.
- [ ] Chạy smoke test 100/200 docs và kiểm tra:
  - [ ] Không có lỗi import/entrypoint.
  - [ ] Trung bình số triples/doc hợp lý.
  - [ ] Có file `cluster_*`, `classify_*`.
  - [ ] Inspect thủ công 10 triples để xem dependency extraction có nhiễu quá nhiều không.

Nhận xét phase hiện tại: đang đi đúng hướng và bám Section 3.1-3.3. Các chỉnh sửa cần thiết đã được bổ sung: entrypoint `arxiv_triples_pipeline.py`, default input trỏ về `dataset/`, thêm `kg_edges`, và cho phép chạy module `data_processing.pipeline` trực tiếp.

## Phase 2 - Embedding generation

- [x] Cài dependencies:
  - [x] `sentence-transformers`
  - [x] `transformers`
  - [x] `torch`
  - [x] `scikit-learn`
  - [x] `hdbscan`
  - [x] `optuna`
  - [x] `evaluate` hoặc tự tính metrics bằng sklearn.
- [x] Tạo script `embedding_pipeline.py`.
- [x] Với mỗi split và representation, encode bằng 4 model:
  - [x] `sentence-transformers/all-MiniLM-L6-v2`
  - [x] `sentence-transformers/all-mpnet-base-v2`
  - [x] `allenai/specter`
  - [x] `allenai/scibert_scivocab_uncased`
- [x] L2-normalize toàn bộ embeddings.
- [x] Lưu embeddings theo format `.npy` hoặc `.parquet`, kèm file metadata mapping `id -> label`.
- [x] Với SciBERT/SPECTER nếu không dùng sentence-transformers trực tiếp:
  - [x] Tokenize max length phù hợp.
  - [x] Mean pooling hoặc CLS pooling phải được ghi rõ.
  - [x] Dùng cùng pooling cho mọi representation.
- [x] Verification:
  - [x] Shape đúng: `n_docs x embedding_dim`.
  - [x] Không có NaN/Inf.
  - [x] Norm vector xấp xỉ 1 sau normalization.

## Phase 3 - Unsupervised clustering

- [x] Tạo script `clustering_pipeline.py`.
- [x] Với mỗi combination `representation x embedding_model`, chạy:
  - [x] KMeans với `k in [3, 12]`.
  - [x] GMM với `k in [3, 12]`.
  - [x] HDBSCAN với sweep `min_cluster_size`.
- [x] Tính metrics:
  - [x] ARI.
  - [x] NMI.
  - [x] Silhouette.
  - [x] Noise fraction cho HDBSCAN.
- [x] Chọn best KMeans/GMM bằng `0.5 * ARI + 0.5 * NMI`.
- [x] Chọn best HDBSCAN bằng `NMI + 0.5 * ARI - 0.5 * noise_fraction`.
- [x] Lưu bảng kết quả tương đương Table 1.
- [x] Lưu cluster label cho 5.000 clustering docs.
- [x] Phân tích label distribution trong cluster:
  - [x] astro-ph cluster.
  - [x] math cluster.
  - [x] cond-mat vs quant-ph.
  - [x] hep/ph/gr-qc overlap.
  - [x] mixed interdisciplinary cluster.
- [x] Expected reproduction target:
  - [x] Abstract + MPNet KMeans/GMM tốt nhất, ARI khoảng `0.47`, NMI khoảng `0.55`.
  - [x] Triples-only thấp hơn rõ.
  - [x] HDBSCAN kém do nhiều noise.

## Phase 4 - Cluster propagation sang classification set

- [x] Dùng nearest-neighbor trong embedding space để gán cluster signal từ 5.000 clustering docs sang 10.000 classification docs.
- [x] Chạy propagation riêng theo từng representation dùng ở clustering stage.
- [x] Lưu `cluster_id` hoặc feature phụ cho từng classification document.
- [x] Kiểm tra phân phối `cluster_id` không bị collapse vào 1-2 cluster.
- [x] Ghi rõ assumption: paper nhắc propagation labels nhưng không mô tả toàn bộ chi tiết implement, nên cần báo cáo cách mình hiện thực.

## Phase 5 - Supervised classification

- [ ] Tạo script `classification_pipeline.py`.
- [ ] Fine-tune `AutoModelForSequenceClassification`.
- [ ] Input text max length `128`.
- [ ] Stratified split `80/20`.
- [ ] Optuna search:
  - [ ] learning rate `[1e-6, 1e-4]`.
  - [ ] batch size `{8, 16, 32}`.
  - [ ] epochs `2-7`.
- [ ] Early stopping theo validation loss.
- [ ] Models cần chạy theo paper:
  - [ ] SciBERT.
  - [ ] SPECTER.
  - [ ] MiniLM nếu class head/pooling được implement ổn định.
- [ ] Chạy 16 experiment chính theo Table 2:
  - [ ] clustering mode: `Abs`, `Trip`, `Abs Trip`, `Hyb`.
  - [ ] classifier input mode: `Abs`, `Trip`, `Abs Trip`, `Hyb`.
- [ ] Metrics:
  - [ ] accuracy.
  - [ ] macro/weighted precision.
  - [ ] macro/weighted recall.
  - [ ] macro/weighted F1.
  - [ ] Cohen's kappa.
  - [ ] MCC.
  - [ ] top-3 accuracy.
  - [ ] macro ROC-AUC one-vs-rest.
- [ ] Expected reproduction target:
  - [ ] Best row gần `Abs/Hyb` với SciBERT.
  - [ ] Accuracy khoảng `92.6%`.
  - [ ] Macro-F1 khoảng `0.925`.
  - [ ] Triples-only thấp hơn abstract/hybrid.

## Phase 6 - Reporting and reproducibility

- [ ] Tạo notebook hoặc markdown report:
  - [ ] dataset/filter mô tả rõ.
  - [ ] triple extraction examples.
  - [ ] Table 1 reproduction.
  - [ ] Table 2 reproduction.
  - [ ] so sánh expected vs actual.
  - [ ] phân tích vì sao lệch nếu kết quả không trùng paper.
- [ ] Lưu confusion matrix cho best classifier.
- [ ] Lưu top errors: abstract, triples, true label, predicted label.
- [ ] Lưu cluster purity examples.
- [ ] Tạo script `run_smoke.ps1` hoặc `Makefile` cho small run.

## Phase 7 - Hướng phát triển NCKH mới: Triple-quality-aware hybrid classification

Ý tưởng: paper chứng minh hybrid tốt hơn, nhưng chưa phân biệt triple tốt/xấu. Hướng mới là làm mô hình biết độ tin cậy của triples và giảm ảnh hưởng của triples nhiễu.

- [ ] Tạo triple quality features:
  - [ ] dependency rule score: direct object cao hơn prepositional fallback.
  - [ ] phrase length score: subject/object quá ngắn hoặc quá dài bị giảm.
  - [ ] relation frequency score: relation quá chung như `be`, `have`, `use` bị giảm.
  - [ ] source sentence confidence: sentence quá dài/nhiều ký hiệu LaTeX bị giảm.
- [ ] Tạo filtered triples:
  - [ ] top-k triples/document theo quality score.
  - [ ] threshold variants: keep all, keep top 50%, keep top 5.
- [ ] Tạo weighted hybrid representation:
  - [ ] `abstract [SEP] high_conf_triples [SEP] low_conf_triples`.
  - [ ] hoặc chỉ dùng high-confidence triples.
- [ ] So sánh với paper baseline:
  - [ ] Abstract.
  - [ ] Triples all.
  - [ ] Abstract+Triples all.
  - [ ] Hybrid all.
  - [ ] Hybrid top-k quality triples.
- [ ] Hypothesis:
  - [ ] Triple-only vẫn thấp hơn abstract vì thiếu ngữ cảnh.
  - [ ] Hybrid top-k có thể tốt hơn Hybrid all nếu triple extraction nhiễu.
  - [ ] Lợi ích rõ nhất ở SciBERT do model xử lý structured scientific text tốt.
- [ ] Metrics chính:
  - [ ] accuracy.
  - [ ] macro-F1.
  - [ ] top-3 accuracy.
  - [ ] per-label F1 gain/loss.
- [ ] Kết quả NCKH mong muốn:
  - [ ] Nếu Hybrid top-k tăng macro-F1 từ `0.925` lên dù chỉ `0.3-0.8` điểm %, đây là đóng góp thực nghiệm hợp lý.
  - [ ] Nếu không tăng, vẫn có đóng góp phân tích: chứng minh triple filtering không đủ, cần extractor tốt hơn hoặc graph-aware model.

## Phase 8 - Hướng phát triển nâng cao nếu còn thời gian: Graph-aware document representation

- [ ] Build global document concept graph từ `kg_edges`.
- [ ] Node là normalized subject/object phrase, edge là relation.
- [ ] Tạo graph features:
  - [ ] degree/top concepts per document.
  - [ ] relation distribution.
  - [ ] entity overlap giữa documents.
- [ ] Thử GraphSAGE/GCN hoặc đơn giản hơn: concatenate transformer embedding với graph feature vector.
- [ ] So sánh:
  - [ ] text-only embedding.
  - [ ] hybrid text+triple embedding.
  - [ ] hybrid + graph features.
- [ ] Đây là extension mạnh hơn nhưng tốn công hơn Phase 7.

## Definition of done

- [ ] Repo có script chạy được cho từng phase.
- [ ] Có smoke test nhỏ hoàn tất dưới vài phút.
- [ ] Có full experiment hoặc subset đủ lớn để báo cáo.
- [ ] Có bảng kết quả tương đương Table 1 và Table 2.
- [ ] Có ít nhất một extension tự phát triển được đánh giá bằng cùng metrics.
- [ ] Báo cáo nêu rõ phần nào reproduce đúng paper, phần nào là assumption, phần nào là đóng góp mới.
