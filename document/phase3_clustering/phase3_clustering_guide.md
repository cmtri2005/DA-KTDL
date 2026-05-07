# Phase 3 - Unsupervised Clustering Guide

Tài liệu này phân rã chi tiết Phase 3 của quá trình tái hiện paper *Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents*. Mục tiêu không phải chỉ là chạy vài thuật toán clustering cho xong, mà là hiểu Phase 3 đang kiểm tra điều gì trong toàn bộ lập luận của paper: embedding nào tạo ra không gian tài liệu khoa học có cấu trúc chủ đề rõ nhất, và việc thêm triples có thật sự giúp hay không.

## 1. Phase 3 đứng ở đâu trong toàn pipeline

Phase 1 đã chuẩn bị dữ liệu arXiv, trích xuất triples từ abstract, tạo document-level knowledge graph edges, và xuất 4 dạng biểu diễn văn bản:

| Tên trong repo | Ý nghĩa | Tên tương ứng trong paper |
| --- | --- | --- |
| `abstract` | Chỉ dùng abstract đã clean | Abstract |
| `triples` | Chỉ dùng các triples đã linearize | Triples |
| `concatenate` | Nối phẳng abstract và triples | Abstract+Triples |
| `hybrid` | `abstract [SEP] triples` | Hybrid |

Phase 2 đã nhận 4 representation đó và encode thành embedding bằng các model như MiniLM, MPNet, SPECTER, SciBERT. Mỗi thí nghiệm Phase 2 lưu theo cấu trúc:

```text
outputs/phase2_embeddings/{split}/{representation}/{model_slug}/
  embeddings.npy
  metadata.jsonl
  verification.json
  run_config.json
```

Phase 3 bắt đầu từ đây. Nó không trích triples lại, không clean abstract lại, và không encode văn bản lại. Nó chỉ lấy embeddings đã sinh ở Phase 2, chạy clustering trên `cluster` split 5.000 documents, rồi đo xem cụm sinh ra có khớp với nhãn chủ đề arXiv hay không.

Điểm cần nhớ: Phase 3 là phép kiểm tra chất lượng không gian embedding. Nếu embedding tốt, các paper cùng lĩnh vực sẽ có xu hướng nằm gần nhau và rơi vào cùng cụm. Nếu embedding kém hoặc representation làm mất ngữ cảnh, cụm sẽ lẫn nhiều nhãn.

## 2. Mục tiêu thật sự của Phase 3

Mục tiêu bề mặt là tạo bảng kết quả tương đương Table 1 trong paper. Nhưng mục tiêu nghiên cứu sâu hơn là trả lời 3 câu hỏi:

1. **Representation nào giúp clustering tốt nhất?**
   - `abstract` giữ đầy đủ ngữ cảnh.
   - `triples` giữ thông tin có cấu trúc nhưng có thể mất nhiều chi tiết ngôn ngữ.
   - `concatenate` kiểm tra việc thêm triples vào cuối abstract có giúp không.
   - `hybrid` kiểm tra việc phân tách rõ abstract và triples bằng `[SEP]` có tạo tín hiệu tốt hơn không.

2. **Embedding model nào tạo không gian tài liệu dễ phân cụm nhất?**
   - MiniLM và MPNet là sentence embedding models mạnh cho semantic similarity.
   - SPECTER và SciBERT được huấn luyện trên ngữ cảnh khoa học hơn, nhưng chưa chắc luôn tốt hơn cho clustering không giám sát.

3. **Thuật toán clustering nào phù hợp với embedding space này?**
   - KMeans giả định cụm xoay quanh centroid.
   - GMM giả định cụm là các phân phối xác suất.
   - HDBSCAN tìm cụm theo mật độ và có thể gán một phần điểm là noise.

Paper dùng clustering để kiểm tra xem các biểu diễn có phản ánh cấu trúc chủ đề khoa học hay không. Vì vậy khi làm Phase 3, không nên chỉ nhìn dòng best score. Cần đọc cả pattern: triples-only thấp hơn abstract ở đâu, hybrid có cải thiện không, HDBSCAN noise nhiều không, và các nhãn liên ngành bị trộn như thế nào.

## 3. Input cần có trước khi làm

Phase 3 chỉ dùng split:

```text
outputs/phase2_embeddings/cluster/
```

Không dùng `classify` split ở Phase 3. Split `classify` được để dành cho các phase sau, đặc biệt là propagation và supervised classification. Việc tách riêng như vậy giúp thí nghiệm clustering và classification không bị overlap dữ liệu, bám với thiết kế trong paper.

Với mỗi combination `representation x embedding_model`, cần có đủ 4 file:

| File | Dùng để làm gì | Vì sao cần |
| --- | --- | --- |
| `embeddings.npy` | Ma trận vector `n_docs x embedding_dim` | Đây là input trực tiếp cho clustering |
| `metadata.jsonl` | Mapping `row_index`, `id`, `label`, `primary_category`, `n_triples` | Cần nhãn thật để tính ARI/NMI và phân tích cụm |
| `verification.json` | Kiểm tra shape, NaN/Inf, norm vector | Tránh chạy clustering trên embedding lỗi |
| `run_config.json` | Ghi model, representation, pooling, normalization | Giúp truy vết kết quả về đúng cấu hình |

Trước khi chạy clustering, nên kiểm tra:

- Số dòng metadata bằng số vector trong `embeddings.npy`.
- `verification.json` không có `has_nan` hoặc `has_inf`.
- Nếu Phase 2 bật L2-normalize, `approx_unit_norm` nên là `true`.
- Label trong `metadata.jsonl` không rỗng.

Nếu bỏ qua bước kiểm tra này, kết quả clustering có thể sai mà không báo lỗi rõ ràng. Ví dụ metadata lệch thứ tự với embedding thì ARI/NMI vẫn tính được, nhưng kết quả không còn ý nghĩa.

## 4. Các thí nghiệm phải chạy

Phase 3 cần chạy toàn bộ ma trận thí nghiệm:

```text
4 representations x 4 embedding models x 3 clustering algorithms
```

Representations:

```text
abstract
triples
concatenate
hybrid
```

Embedding models:

```text
minilm
mpnet
specter
scibert
```

Algorithms:

```text
KMeans
GMM
HDBSCAN
```

### 4.1 KMeans

Chạy KMeans với:

```text
k = 3..12
random_state = 42
```

Vì sao cần làm:

- KMeans là baseline clustering phổ biến nhất cho embedding.
- Paper dùng KMeans để tạo kết quả so sánh trong Table 1.
- Việc sweep `k = 3..12` giúp không giả định trước số cụm tối ưu. Nhãn arXiv có thể nhiều hơn hoặc ít hơn số nhóm chủ đề tự nhiên trong embedding.

Điểm cần hiểu:

- `k` không nhất thiết bằng số label thật.
- KMeans luôn gán mọi document vào một cụm, nên không có noise.
- Nếu `abstract + MPNet` tốt, thường KMeans sẽ cho ARI/NMI cao hơn triples-only.

Mức độ bám paper:

- Bám sát ở việc dùng KMeans cho clustering.
- Sweep `k = 3..12` là quyết định hiện thực hóa trong repo để tìm cấu hình tốt nhất một cách nhất quán.

### 4.2 GMM

Chạy Gaussian Mixture Model với:

```text
k = 3..12
random_state = 42
```

Vì sao cần làm:

- GMM là cách nhìn mềm hơn KMeans: mỗi cụm là một phân phối Gaussian.
- Nó kiểm tra xem embedding space có phù hợp với giả định cụm dạng phân phối hay không.
- Paper có dùng GMM, nên cần chạy để tái hiện Table 1.

Điểm cần hiểu:

- GMM vẫn xuất ra một cluster label cuối cùng cho mỗi document.
- GMM có thể tốt hơn KMeans nếu cụm có hình dạng/độ lệch khác nhau.
- Nhưng trong embedding space nhiều chiều, GMM cũng có thể không ổn định nếu dữ liệu không giống Gaussian.

Mức độ bám paper:

- Bám sát ở thuật toán và mục tiêu so sánh.
- Cách chọn best theo điểm tổng hợp là quyết định repo, cần ghi rõ trong report.

### 4.3 HDBSCAN

Chạy HDBSCAN với sweep `min_cluster_size`, ví dụ:

```text
min_cluster_size in [5, 10, 15, 25, 50, 100]
```

Có thể giữ mặc định hoặc cố định thêm:

```text
metric = "euclidean"
min_samples = None
```

Vì sao cần làm:

- HDBSCAN không cần định trước số cụm.
- Nó có khả năng đánh dấu các document không thuộc cụm rõ ràng là noise với label `-1`.
- Paper ghi nhận HDBSCAN thường kém trong không gian embedding này vì có nhiều noise hoặc cụm không tách theo mật độ rõ.

Điểm cần hiểu:

- HDBSCAN không chỉ cần ARI/NMI; phải xem thêm `noise_fraction`.
- Nếu noise quá cao, Silhouette/ARI/NMI có thể khó diễn giải.
- Một kết quả HDBSCAN có NMI vừa phải nhưng noise 60-80% không phải là clustering tốt cho mục tiêu tái hiện paper.

Mức độ bám paper:

- Bám sát ở việc dùng HDBSCAN để so sánh với KMeans/GMM.
- Sweep `min_cluster_size` và công thức chọn best là quyết định hiện thực hóa thêm, vì paper không mô tả đủ chi tiết để tái tạo y hệt.

## 5. Metrics cần tính và cách hiểu

### ARI - Adjusted Rand Index

ARI đo mức độ cặp document được gom cùng/khác cụm có khớp với nhãn thật hay không, đã hiệu chỉnh yếu tố ngẫu nhiên.

Vì sao cần:

- Đây là metric chuẩn cho clustering khi có ground-truth labels.
- Nó giúp biết cluster label có thật sự phản ánh category arXiv không.

Cách đọc:

- Gần `1.0`: cụm rất khớp nhãn thật.
- Gần `0.0`: không tốt hơn ngẫu nhiên nhiều.
- Âm: tệ hơn random trong một số trường hợp.

### NMI - Normalized Mutual Information

NMI đo lượng thông tin chung giữa cluster assignment và label thật.

Vì sao cần:

- Nó ít khắt khe hơn ARI trong một số tình huống cụm bị chia nhỏ.
- Paper dùng NMI để đánh giá clustering.

Cách đọc:

- Cao nghĩa là biết cluster giúp đoán label tốt hơn.
- Nếu NMI cao nhưng ARI thấp, có thể cụm giữ thông tin nhãn nhưng bị split/merge chưa đúng.

### Silhouette

Silhouette đo một điểm gần các điểm cùng cụm hơn các điểm khác cụm đến mức nào.

Vì sao cần:

- Đây là metric nội tại, không cần label thật.
- Nó cho biết cụm có tách hình học rõ trong embedding space không.

Cách đọc:

- Cao hơn thường tốt hơn.
- Nhưng không nên dùng một mình, vì cụm hình học đẹp chưa chắc khớp label arXiv.

### Noise fraction

Noise fraction chỉ áp dụng cho HDBSCAN:

```text
noise_fraction = số document có cluster_id = -1 / tổng số document
```

Vì sao cần:

- HDBSCAN có thể đạt score nhìn được trên phần dữ liệu còn lại nhưng bỏ quá nhiều document thành noise.
- Paper nhận xét HDBSCAN kém một phần vì vấn đề density/noise.

Cách đọc:

- Thấp: HDBSCAN gom được phần lớn dữ liệu.
- Cao: nhiều document không thuộc cụm nào, kết quả khó dùng cho propagation/classification sau này.

## 6. Cách chọn best run

Với KMeans và GMM, chọn best config theo:

```text
score = 0.5 * ARI + 0.5 * NMI
```

Lý do:

- ARI và NMI đều là external metrics có ground-truth label.
- ARI khắt khe về cấu trúc cặp document.
- NMI đo lượng thông tin label được giữ lại.
- Trung bình hai metric giúp tránh chọn run chỉ tốt theo một phía.

Với HDBSCAN, chọn best config theo:

```text
score = NMI + 0.5 * ARI - 0.5 * noise_fraction
```

Lý do:

- HDBSCAN cần bị phạt nếu bỏ quá nhiều điểm thành noise.
- NMI được giữ trọng số chính vì HDBSCAN có thể tạo số cụm tự do, dễ split/merge khác số nhãn thật.
- ARI vẫn được tính để kiểm tra độ khớp cụm-label.

Mức độ bám paper:

- Paper yêu cầu so sánh clustering bằng các metrics như ARI/NMI/Silhouette.
- Công thức chọn best không phải chi tiết chắc chắn từ paper; đây là rule thực nghiệm của repo để có kết quả nhất quán và giải thích được.
- Khi viết report, cần ghi rõ đây là implementation assumption.

## 7. Output cần sinh ra

Phase 3 nên tạo output dưới:

```text
outputs/phase3_clustering/
```

Các file nên có:

| Output | Nội dung | Vì sao cần |
| --- | --- | --- |
| `clustering_results.csv` | Tất cả run: representation, model, algorithm, params, ARI, NMI, Silhouette, noise_fraction, score | Là bảng tổng hợp để lọc/sort và tạo Table 1 |
| `table1_reproduction.csv` | Best result theo từng representation/model/algorithm | Bảng gần với Table 1 nhất |
| `best_configs.json` | Config tốt nhất cho mỗi nhóm thí nghiệm | Giúp chạy lại hoặc dùng cho Phase 4 |
| `cluster_labels/*.jsonl` hoặc `.csv` | Cluster id cho từng document | Cần cho phân tích cụm và propagation |
| `cluster_label_distribution/*.csv` | Phân phối label thật trong từng cluster | Giúp đọc chất lượng cụm, không chỉ nhìn metric |
| `run_config.json` | Config chung của Phase 3 | Đảm bảo reproducibility |

Cluster label file nên giữ tối thiểu:

```text
row_index
id
label
primary_category
representation
embedding_model
algorithm
cluster_id
```

Với HDBSCAN, `cluster_id = -1` nghĩa là noise.

## 8. Phân tích kết quả để không làm máy móc

Sau khi có bảng metric, không nên dừng ở câu “model A cao nhất”. Cần đọc kết quả theo logic của paper.

### 8.1 So sánh `abstract` với `triples`

Câu hỏi:

- Triples-only có giữ đủ ngữ nghĩa để gom tài liệu cùng chủ đề không?
- Nếu triples-only thấp hơn rõ, có phải vì triples mất ngữ cảnh, extractor nhiễu, hoặc số triples/doc quá ít?

Kỳ vọng theo paper:

- `triples` thường thấp hơn `abstract`.
- Điều này không phủ định giá trị của triples; nó cho thấy triples không nên thay thế abstract hoàn toàn.

### 8.2 So sánh `concatenate`/`hybrid` với `abstract`

Câu hỏi:

- Khi thêm triples vào abstract, clustering có tốt hơn không?
- `[SEP]` trong `hybrid` có giúp model phân biệt văn bản gốc và tri thức có cấu trúc không?

Kỳ vọng theo paper:

- Trong clustering, abstract thường vẫn rất mạnh.
- Hybrid có thể không thắng rõ ở clustering, nhưng có vai trò quan trọng hơn ở classification.

Điểm cần ghi trong phân tích:

- Nếu `hybrid` không tốt hơn `abstract`, không có nghĩa Phase 1 triples vô ích.
- Có thể clustering không tận dụng triples tốt bằng supervised classifier.
- Cũng có thể triples extractor hiện tại còn nhiễu.

### 8.3 So sánh model embedding

Câu hỏi:

- MiniLM/MPNet có tốt hơn SciBERT/SPECTER trong clustering không?
- Nếu model khoa học không thắng, lý do có thể là gì?

Kỳ vọng theo paper:

- MPNet hoặc MiniLM có thể cho clustering tốt nhất.
- SciBERT/SPECTER mạnh trong ngữ cảnh scientific NLP, nhưng embedding pooling đơn giản chưa chắc tối ưu cho clustering.

### 8.4 Đọc label distribution trong cluster

Với mỗi best run, nên mở vài cluster lớn và xem:

- Cluster nào gần như thuần một label?
- Cluster nào bị trộn nhiều label?
- Label nào hay bị trộn với nhau?

Các nhóm nên kiểm tra:

- `astro-ph` cluster: có tách rõ không?
- `math` cluster: có bị trộn với `cs` hoặc `stat` không?
- `cond-mat` và `quant-ph`: có bị gom chung vì gần vật lý lượng tử không?
- `hep-*`, `physics`, `gr-qc`: có overlap do cùng miền vật lý năng lượng cao/tương đối rộng không?
- Mixed interdisciplinary cluster: có chứa các paper giao thoa nhiều lĩnh vực không?

Phần này giúp bạn đọc kết quả như một nhà nghiên cứu. ARI/NMI cho biết mức độ tổng quát, nhưng label distribution cho biết cụm sai ở đâu và vì sao sai.

## 9. Expected reproduction target

Khi chạy full experiment, kết quả không nhất thiết trùng paper tuyệt đối vì có nhiều chi tiết paper không công bố đầy đủ, ví dụ mốc "recent years", snapshot dataset, exact preprocessing, pooling, hyperparameters. Nhưng pattern kỳ vọng là:

- `abstract + MPNet + KMeans/GMM` nằm trong nhóm tốt nhất.
- ARI tốt nhất có thể quanh `0.47`.
- NMI tốt nhất có thể quanh `0.55`.
- `triples` thấp hơn `abstract` rõ rệt.
- HDBSCAN thường kém hơn KMeans/GMM và có noise fraction đáng kể.
- `concatenate`/`hybrid` cần được so với `abstract`, nhưng không mặc định sẽ thắng trong clustering.

Nếu kết quả lệch nhiều, nên kiểm tra theo thứ tự:

1. Phase 1 có đúng 5.000 clustering docs không.
2. Label mapping có lấy category đầu tiên và map về nhóm lớn như `cs`, `math`, `physics` không.
3. Embeddings có được L2-normalize không.
4. Metadata có khớp thứ tự với embeddings không.
5. Pooling của SciBERT/SPECTER có nhất quán không.
6. Dataset filter theo năm có khác paper quá nhiều không.

## 10. Độ bám sát paper và assumption của repo

### Bám sát paper

Phase 3 bám sát paper ở các điểm:

- Dùng 4 representation: Abstract, Triples, Abstract+Triples, Hybrid.
- Dùng embedding từ MiniLM, MPNet, SPECTER, SciBERT.
- Chạy clustering không giám sát bằng KMeans, GMM, HDBSCAN.
- Đánh giá bằng ARI, NMI, Silhouette.
- Tạo bảng kết quả tương đương Table 1.
- Phân tích xem triples/knowledge-infused representation có giúp clustering scientific documents không.

### Assumption hoặc hiện thực hóa thêm

Các điểm sau là quyết định của repo, cần ghi rõ khi báo cáo:

- Repo dùng tên `concatenate` cho representation paper gọi là `Abstract+Triples`.
- KMeans/GMM sweep `k = 3..12`.
- HDBSCAN sweep `min_cluster_size`.
- Công thức chọn best KMeans/GMM: `0.5 * ARI + 0.5 * NMI`.
- Công thức chọn best HDBSCAN: `NMI + 0.5 * ARI - 0.5 * noise_fraction`.
- Output format cụ thể dưới `outputs/phase3_clustering/`.
- Nếu dataset có lọc năm, mốc năm là assumption vì paper chỉ nói "recent years" nhưng không công bố chính xác.

Những assumption này không làm Phase 3 lệch hướng. Ngược lại, chúng giúp thí nghiệm có thể chạy lại, kiểm tra lại, và giải thích rõ hơn.

## 11. Checklist làm Phase 3

Trước khi chạy:

- [ ] Có Phase 2 embeddings cho split `cluster`.
- [ ] Có đủ 4 representations: `abstract`, `triples`, `concatenate`, `hybrid`.
- [ ] Có đủ 4 embedding models hoặc ghi rõ model nào đang thiếu.
- [ ] `verification.json` của từng job không lỗi NaN/Inf.
- [ ] Metadata và embeddings có cùng số dòng.

Khi chạy:

- [ ] Chạy KMeans với `k = 3..12`.
- [ ] Chạy GMM với `k = 3..12`.
- [ ] Chạy HDBSCAN với sweep `min_cluster_size`.
- [ ] Tính ARI, NMI, Silhouette cho mọi run hợp lệ.
- [ ] Tính noise fraction cho HDBSCAN.
- [ ] Chọn best run theo rule đã định.

Sau khi chạy:

- [ ] Lưu `clustering_results.csv`.
- [ ] Lưu bảng best tương đương Table 1.
- [ ] Lưu cluster labels cho 5.000 documents.
- [ ] Lưu phân phối label trong từng cluster.
- [ ] Viết nhận xét: representation nào tốt, model nào tốt, thuật toán nào kém, và pattern có giống paper không.

## 12. Cách tự đọc kết quả

Khi có kết quả, hãy đọc theo thứ tự này:

1. Nhìn best ARI/NMI tổng thể để biết cấu hình mạnh nhất.
2. So sánh theo representation để trả lời vai trò của triples.
3. So sánh theo embedding model để biết model nào tạo không gian tốt.
4. So sánh KMeans/GMM/HDBSCAN để biết giả định clustering nào phù hợp.
5. Mở label distribution của vài cluster lớn để hiểu lỗi cụ thể.
6. Đối chiếu pattern với paper thay vì chỉ đối chiếu từng con số.

Câu kết luận tốt cho Phase 3 không nên chỉ là:

```text
MPNet KMeans cao nhất.
```

Mà nên là:

```text
Trong clustering không giám sát, abstract embedding vẫn giữ cấu trúc chủ đề tốt hơn triples-only. Việc thêm triples chưa chắc cải thiện rõ ở clustering, nhưng kết quả này phù hợp với nhận xét của paper rằng triples riêng lẻ thiếu ngữ cảnh. KMeans/GMM ổn định hơn HDBSCAN, còn HDBSCAN tạo nhiều noise trong embedding space nhiều chiều.
```

Đó là cách Phase 3 giúp mình hiểu paper, thay vì chỉ tái hiện máy móc.
