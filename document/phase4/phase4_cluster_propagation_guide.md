# Phase 4 - Cluster Propagation sang Classification Set

Tài liệu này mô tả những việc cần làm trong Phase 4 của quá trình tái hiện paper *Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents*. Mục tiêu chính là giúp các thành viên trong nhóm hiểu Phase 4 dùng để làm gì, vì sao cần có phase này, input/output cần chuẩn bị ra sao, và khi implement cần kiểm tra những điểm nào.

## 1. Phase 4 đứng ở đâu trong toàn pipeline

Trước Phase 4, repo đã có 3 phần chính:

| Phase | Vai trò | Output chính |
| --- | --- | --- |
| Phase 1 | Chuẩn bị dữ liệu, trích triples, tạo 4 representation | `outputs/phase1_data/` |
| Phase 2 | Encode các representation thành embeddings | `outputs/phase2_embeddings/` |
| Phase 3 | Chạy clustering trên 5.000 documents của `cluster` split | `outputs/phase3_clustering/` |

Phase 4 nằm giữa clustering và supervised classification. Phase này nhận cluster labels đã sinh ở Phase 3 trên `cluster` split, sau đó lan truyền cluster signal sang 10.000 documents của `classify` split.

Nói ngắn gọn:

```text
cluster docs có cluster_id
        +
classify docs chưa có cluster_id
        |
        v
nearest-neighbor propagation trong embedding space
        |
        v
classify docs có thêm propagated_cluster_id
```

Sau Phase 4, mỗi document trong classification set sẽ có thêm một hoặc nhiều feature liên quan tới cluster. Feature này có thể được dùng trong Phase 5 để kiểm tra xem thông tin cụm học được từ không gian embedding có giúp bài toán phân loại hay không.

## 2. Vì sao cần Phase 4

Paper không chỉ quan tâm đến clustering và classification như hai bài toán rời rạc. Ý tưởng của paper là triples và knowledge-infused embeddings có thể tạo ra cấu trúc chủ đề tốt hơn trong embedding space. Nếu cấu trúc đó có ý nghĩa, cluster signal từ các tài liệu đã clustering có thể trở thành thông tin phụ trợ cho classification.

Vì vậy Phase 4 có 3 lý do tồn tại:

1. **Kết nối unsupervised clustering với supervised classification**

   Phase 3 tạo ra cluster labels nhưng chỉ trên 5.000 documents dành riêng cho clustering. Phase 5 lại train/evaluate trên 10.000 documents classification. Nếu muốn dùng kết quả clustering như một signal trong classification, ta cần một cách gán cluster cho các classification documents mà không phá vỡ split ban đầu.

2. **Giữ thiết kế non-overlap giữa clustering và classification**

   Phase 1 đã chia dữ liệu thành 2 split không overlap. Không nên chạy lại clustering trực tiếp trên toàn bộ classification set nếu mục tiêu là giữ đúng tinh thần experimental setup. Propagation giúp tận dụng cluster labels từ cluster split mà không trộn hai tập dữ liệu vào cùng một bước clustering.

3. **Tạo feature phụ có thể kiểm chứng được**

   Nếu document classification gần các document thuộc một cluster nào đó trong embedding space, cluster đó có thể là tín hiệu về chủ đề ngữ nghĩa. Phase 5 có thể dùng tín hiệu này theo nhiều cách: thêm `cluster_id`, thêm one-hot cluster vector, thêm phân phối neighbor clusters, hoặc dùng để tổ chức các experiment theo clustering mode.

Điểm quan trọng: paper có nhắc tới propagation labels nhưng không mô tả đầy đủ chi tiết implement. Vì vậy Phase 4 trong repo cần ghi rõ assumption và cách hiện thực hóa để kết quả có thể được đọc, kiểm tra, và tái chạy.

## 3. Input cần có

Phase 4 cần input từ Phase 2 và Phase 3.

### 3.1 Embeddings của cluster split

Với mỗi representation và embedding model, cần có:

```text
outputs/phase2_embeddings/cluster/{representation}/{model_slug}/
  embeddings.npy
  metadata.jsonl
  verification.json
  run_config.json
```

Những file này cho biết vector embedding và metadata của 5.000 documents đã được clustering.

### 3.2 Embeddings của classify split

Với cùng representation và cùng embedding model, cần có:

```text
outputs/phase2_embeddings/classify/{representation}/{model_slug}/
  embeddings.npy
  metadata.jsonl
  verification.json
  run_config.json
```

Đây là 10.000 documents sẽ được gán cluster signal.

### 3.3 Cluster labels từ Phase 3

Phase 3 đã lưu cluster labels theo từng job, ví dụ:

```text
outputs/phase3_clustering/cluster/{representation}/{model_slug}/
  kmeans_k*_labels.npy
  gmm_k*_labels.npy
  hdbscan_mcs*_labels.npy
  phase2_job.json
  doc_ids.txt
```

Ngoài ra Phase 3 có các bảng tổng hợp:

```text
outputs/phase3_clustering/results_table.csv
outputs/phase3_clustering/results_table_best_by_algorithm.csv
outputs/phase3_clustering/results_table_best_by_representation.csv
```

Phase 4 nên dùng cấu hình clustering đã được chọn là tốt nhất từ Phase 3, thường là best theo từng `representation x model x algorithm`, hoặc một cấu hình cố định được nhóm thống nhất để tái hiện paper.

## 4. Output cần tạo

Output Phase 4 nên nằm trong:

```text
outputs/phase4_cluster_propagation/
```

Với mỗi propagation job, nên lưu:

```text
outputs/phase4_cluster_propagation/{representation}/{model_slug}/{algorithm_or_cluster_config}/
  propagated_clusters.jsonl
  propagated_clusters.csv
  propagation_config.json
  propagation_summary.json
```

### 4.1 `propagated_clusters.jsonl`

Mỗi dòng tương ứng với một document trong classification set. Các field nên có:

| Field | Ý nghĩa |
| --- | --- |
| `id` | ID document classification |
| `label` | Label thật từ arXiv category, dùng để đánh giá về sau |
| `primary_category` | Category gốc, ví dụ `cs.AI` |
| `representation` | `abstract`, `triples`, `concatenate`, hoặc `hybrid` |
| `embedding_model` | Model tạo embedding |
| `source_cluster_algorithm` | KMeans, GMM, hoặc HDBSCAN |
| `propagated_cluster_id` | Cluster được gán từ nearest neighbors |
| `neighbor_ids` | Danh sách ID các cluster docs gần nhất |
| `neighbor_distances` | Khoảng cách tới các nearest neighbors |
| `neighbor_cluster_ids` | Cluster labels của các neighbors |
| `propagation_confidence` | Điểm tin cậy của propagation |

### 4.2 `propagation_config.json`

File này giúp tái lập experiment. Nên ghi:

| Field | Ví dụ |
| --- | --- |
| `seed` | `42` |
| `representation` | `hybrid` |
| `embedding_model` | `sentence-transformers/all-mpnet-base-v2` |
| `cluster_algorithm` | `kmeans` |
| `cluster_label_file` | path tới file labels từ Phase 3 |
| `neighbor_k` | `5` |
| `distance_metric` | `cosine` hoặc `euclidean` |
| `assignment_rule` | `majority_vote` hoặc `distance_weighted_vote` |
| `handle_noise` | cách xử lý label `-1` của HDBSCAN |

### 4.3 `propagation_summary.json`

File này dùng để kiểm tra nhanh propagation có ổn không. Nên có:

| Field | Ý nghĩa |
| --- | --- |
| `n_classify_docs` | Số document được gán cluster |
| `n_source_cluster_docs` | Số document nguồn từ cluster split |
| `n_unique_propagated_clusters` | Số cluster xuất hiện sau propagation |
| `cluster_distribution` | Phân phối số document theo cluster |
| `largest_cluster_fraction` | Tỷ lệ document rơi vào cluster lớn nhất |
| `noise_fraction` | Tỷ lệ document bị gán noise nếu dùng HDBSCAN |
| `mean_neighbor_distance` | Khoảng cách trung bình tới neighbors |
| `mean_confidence` | Confidence trung bình |

## 5. Cách propagation đề xuất

Vì Phase 2 đã L2-normalize embeddings, có thể dùng cosine distance hoặc euclidean distance. Với vector đã normalize, hai metric này thường cho thứ tự neighbor khá tương đồng. Nhóm nên chọn một metric và ghi rõ trong config.

Quy trình đề xuất:

1. Load embeddings và metadata của `cluster` split.
2. Load embeddings và metadata của `classify` split.
3. Load cluster labels từ Phase 3 cho đúng `representation`, `model_slug`, và thuật toán clustering.
4. Kiểm tra thứ tự `doc_ids.txt` ở Phase 3 khớp với metadata của `cluster` embeddings.
5. Fit nearest-neighbor index trên cluster embeddings.
6. Với mỗi classify embedding, tìm `k` cluster documents gần nhất.
7. Gán `propagated_cluster_id` bằng luật vote từ cluster labels của neighbors.
8. Lưu kết quả và summary.

Pseudo-code:

```text
cluster_embeddings = load cluster embeddings
cluster_metadata = load cluster metadata
cluster_labels = load labels from phase 3

classify_embeddings = load classify embeddings
classify_metadata = load classify metadata

nn = NearestNeighbors(metric="cosine", n_neighbors=5)
nn.fit(cluster_embeddings)

for each classify_doc:
    neighbor_indices, distances = nn.query(classify_embedding)
    neighbor_labels = cluster_labels[neighbor_indices]
    propagated_cluster_id = vote(neighbor_labels, distances)
    confidence = compute_confidence(neighbor_labels, distances)
    save result
```

## 6. Luật gán cluster nên dùng

Có 2 lựa chọn chính.

### 6.1 Majority vote

Lấy `k` neighbors gần nhất, cluster nào xuất hiện nhiều nhất thì gán cluster đó.

Vì sao nên dùng:

- Dễ hiểu, dễ implement, dễ giải thích trong báo cáo.
- Ít nhạy với một neighbor rất gần nhưng có thể là outlier.
- Phù hợp để làm baseline Phase 4.

Điểm cần lưu ý:

- Nếu hòa phiếu, chọn cluster của neighbor gần nhất.
- Nếu dùng HDBSCAN và nhiều neighbor có label `-1`, cần quyết định có cho phép gán `-1` hay bỏ qua noise.

### 6.2 Distance-weighted vote

Mỗi neighbor vote theo trọng số phụ thuộc khoảng cách. Neighbor càng gần thì trọng số càng cao.

Ví dụ:

```text
weight = 1 / (distance + epsilon)
```

Vì sao có thể dùng:

- Tận dụng thông tin khoảng cách tốt hơn majority vote.
- Hợp lý khi neighbor gần nhất rất gần và các neighbor khác xa hơn nhiều.

Điểm cần lưu ý:

- Cần tránh chia cho 0 bằng `epsilon`.
- Kết quả khó giải thích hơn majority vote một chút.
- Nên xem là option mở rộng sau khi majority vote chạy ổn.

Đề xuất cho repo: implement majority vote trước, sau đó thêm distance-weighted vote nếu còn thời gian.

## 7. Chọn `k` nearest neighbors như thế nào

Giá trị `k` nên là một hyperparameter có thể cấu hình. Gợi ý:

```text
k in [1, 3, 5, 10]
```

Đề xuất mặc định:

```text
k = 5
```

Vì sao:

- `k = 1` dễ hiểu nhưng nhạy với outlier.
- `k = 3` hoặc `k = 5` thường cân bằng giữa ổn định và vẫn giữ tính local.
- `k = 10` ổn định hơn nhưng có thể làm mờ ranh giới giữa các cluster gần nhau.

Trong report, nếu có thời gian nên chạy sensitivity nhỏ theo `k` để xem propagation có bị thay đổi mạnh không.

## 8. Xử lý HDBSCAN noise label `-1`

HDBSCAN có thể gán một số cluster documents là noise với label `-1`. Khi propagation, cần quyết định rõ cách xử lý.

Có 3 phương án:

| Phương án | Cách làm | Ưu điểm | Nhược điểm |
| --- | --- | --- | --- |
| Giữ `-1` | Cho phép classify doc nhận `propagated_cluster_id = -1` | Trung thực với HDBSCAN | Có thể tạo quá nhiều noise |
| Bỏ qua `-1` khi vote | Chỉ vote bằng non-noise neighbors | Giảm noise trong classification feature | Nếu toàn bộ neighbors là `-1`, cần fallback |
| Không dùng HDBSCAN cho Phase 4 chính | Chỉ propagate từ KMeans/GMM best | Đơn giản, ổn định hơn | Không khai thác HDBSCAN |

Đề xuất cho baseline: dùng KMeans hoặc GMM best từ Phase 3 cho propagation chính. HDBSCAN có thể giữ như thí nghiệm phụ vì Phase 3 thường cho thấy HDBSCAN kém và có nhiều noise.

## 9. Các kiểm tra bắt buộc

Phase 4 rất dễ sinh output nhìn có vẻ hợp lệ nhưng thực ra sai do lệch thứ tự metadata, nhầm representation, hoặc cluster collapse. Vì vậy cần có các kiểm tra sau.

### 9.1 Kiểm tra alignment

Trước khi dùng labels từ Phase 3, phải kiểm tra:

- Số label bằng số dòng metadata cluster.
- `doc_ids.txt` từ Phase 3 khớp đúng thứ tự với `metadata.jsonl` của cluster embeddings.
- Representation và model trong `phase2_job.json` khớp với input Phase 4.

Vì sao cần:

Nếu thứ tự labels lệch với embeddings, propagation sẽ gán cluster sai mà code vẫn chạy bình thường. Đây là lỗi nguy hiểm nhất của Phase 4.

### 9.2 Kiểm tra phân phối cluster

Sau khi propagate, cần xem:

- Có bao nhiêu cluster xuất hiện.
- Cluster lớn nhất chiếm bao nhiêu phần trăm.
- Có cluster nào nhận 0 document không.
- Nếu dùng HDBSCAN, noise fraction là bao nhiêu.

Vì sao cần:

Nếu 80-90% classification docs bị gán vào 1 cluster, cluster signal gần như không còn ý nghĩa. Điều đó có thể do chọn clustering config kém, dùng sai metric, hoặc embeddings bị collapse.

### 9.3 Kiểm tra khoảng cách neighbor

Nên lưu và xem:

- Mean/median distance tới neighbor gần nhất.
- Mean/median distance tới top-k neighbors.
- Một vài document có distance cao nhất.

Vì sao cần:

Nếu classification docs đều rất xa cluster docs, propagation không đáng tin. Cluster signal lúc đó có thể là nhiễu thay vì thông tin phụ trợ.

### 9.4 Inspect thủ công

Chọn khoảng 10-20 documents để xem:

- Abstract hoặc text representation.
- Label thật.
- Propagated cluster.
- Các neighbors gần nhất.
- Labels và cluster ids của neighbors.

Vì sao cần:

Metrics và distribution chỉ cho biết pattern tổng quát. Inspect thủ công giúp phát hiện lỗi dễ bỏ sót như neighbor không cùng miền chủ đề, triples quá nhiễu, hoặc nhãn cluster khó diễn giải.

## 10. Phase 4 sẽ được dùng trong Phase 5 như thế nào

Phase 5 là supervised classification. Cluster signal từ Phase 4 có thể được dùng theo nhiều mức độ.

### 10.1 Dùng như metadata feature

Thêm `propagated_cluster_id` vào record classification. Khi training classifier truyền thống hoặc head phụ, có thể one-hot encode cluster id rồi concatenate với embedding/text features.

Ưu điểm:

- Rõ ràng, dễ phân tích.
- Có thể đo trực tiếp việc thêm cluster feature có cải thiện không.

### 10.2 Dùng để tổ chức experiment

Paper có ý tưởng so sánh theo clustering mode và classifier input mode. Phase 4 giúp tạo ra các version classification set tương ứng với từng clustering mode:

```text
cluster mode: abstract / triples / concatenate / hybrid
classifier input: abstract / triples / concatenate / hybrid
```

Nhờ vậy Phase 5 có thể chạy ma trận experiment giống Table 2 hơn.

### 10.3 Dùng như tín hiệu phân tích sau huấn luyện

Ngay cả khi không đưa `cluster_id` trực tiếp vào model, Phase 4 vẫn hữu ích để phân tích:

- Model sai nhiều ở cluster nào?
- Cluster nào chứa nhiều label liên ngành?
- Triples/hybrid propagation có làm các nhóm chủ đề rõ hơn abstract-only không?

## 11. Assumptions cần ghi rõ trong báo cáo

Vì paper không mô tả đầy đủ chi tiết label propagation, repo cần ghi rõ các assumption sau:

1. Propagation được thực hiện bằng nearest neighbors trong embedding space.
2. Source set là 5.000 cluster documents đã có cluster labels từ Phase 3.
3. Target set là 10.000 classification documents, không dùng để fit clustering.
4. Propagation chạy riêng cho từng representation và embedding model.
5. Cluster config được chọn từ kết quả Phase 3, ví dụ best KMeans theo `0.5 * ARI + 0.5 * NMI`.
6. `neighbor_k`, distance metric, và assignment rule được ghi trong `propagation_config.json`.
7. Nếu dùng HDBSCAN, cách xử lý noise label `-1` phải được ghi rõ.

Các assumption này không làm thí nghiệm sai. Ngược lại, chúng giúp kết quả minh bạch hơn vì người đọc biết chính xác nhóm đã hiện thực phần paper còn thiếu như thế nào.

## 12. Checklist implement Phase 4

- [ ] Tạo module/script Phase 4, ví dụ `propagation/cluster_propagation_pipeline.py` hoặc `classification/cluster_propagation.py`.
- [ ] CLI nhận các tham số:
  - [ ] `--phase2_root`
  - [ ] `--phase3_root`
  - [ ] `--output_root`
  - [ ] `--representation`
  - [ ] `--model`
  - [ ] `--cluster_algorithm`
  - [ ] `--cluster_label_file`
  - [ ] `--neighbor_k`
  - [ ] `--metric`
  - [ ] `--assignment_rule`
  - [ ] `--seed`
- [ ] Load và validate cluster embeddings, classify embeddings, metadata.
- [ ] Load cluster labels từ Phase 3.
- [ ] Kiểm tra alignment giữa `doc_ids.txt`, metadata, và labels.
- [ ] Fit nearest-neighbor index trên cluster embeddings.
- [ ] Propagate cluster id cho từng classify document.
- [ ] Tính propagation confidence.
- [ ] Lưu `propagated_clusters.jsonl`.
- [ ] Lưu `propagated_clusters.csv`.
- [ ] Lưu `propagation_config.json`.
- [ ] Lưu `propagation_summary.json`.
- [ ] Kiểm tra cluster distribution không collapse vào 1-2 cluster.
- [ ] Inspect thủ công một số propagated examples.

## 13. Definition of done cho Phase 4

Phase 4 được xem là hoàn tất khi:

- Có script chạy được từ CLI.
- Có thể chạy smoke test trên output nhỏ.
- Có thể chạy full propagation cho ít nhất một cấu hình tốt nhất từ Phase 3.
- Output có đủ propagated cluster id cho toàn bộ classification docs.
- Có file config và summary để tái lập.
- Có kiểm tra alignment để tránh gán nhầm cluster labels.
- Báo cáo ghi rõ cách nhóm hiện thực label propagation vì paper không cung cấp đủ chi tiết.

Kết quả mong muốn không phải là cluster signal chắc chắn cải thiện classification ngay lập tức. Kết quả mong muốn của Phase 4 là tạo ra một tầng dữ liệu trung gian đúng, minh bạch, có thể kiểm chứng, để Phase 5 có thể đánh giá nghiêm túc tác động của clustering lên supervised classification.
