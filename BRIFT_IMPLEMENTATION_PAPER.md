# BRIFT Implementation Paper

Tài liệu này giải thích lại pipeline tái hiện paper **Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents** trong repo hiện tại, dựa trên:

- `paper_reimplementation_todo.md`: contract/plan tái hiện paper.
- `Triples.pdf`: bài báo gốc trong workspace. Metadata PDF local trỏ tới arXiv `2601.08841v1`.
- `summary_paper.md`: tóm tắt phương pháp và kết quả chính.
- Source code và artifacts trong các thư mục `data_processing/`, `embeddings/`, `clustering/`, `propagation/`, `classification/`, `reports/`, `outputs/`.

Mục tiêu của file này là giúp người đọc hiểu **mỗi phase làm gì, vì sao phải làm như vậy, phase đó đang bám paper/plan tới đâu, và còn chỗ nào cần cẩn trọng khi báo cáo kết quả**.

> Lưu ý phiên bản: `Triples.pdf` local là v1. Trang arXiv hiện tại có thể có v2 với diễn giải/kết luận khác hơn. Repo này, plan hiện tại và các expected targets trong `paper_reimplementation_todo.md` đang bám theo nội dung v1/summary local, ví dụ kỳ vọng best classification khoảng `92.6%` accuracy và macro-F1 khoảng `0.925`.

---

## 1. Kết luận đánh giá nhanh

Pipeline đã được hiện thực khá đầy đủ về mặt module:

- Phase 1 có code và output thật cho 5.000 document clustering + 10.000 document classification.
- Phase 2 có đủ 32 embedding jobs: 2 split x 4 representation x 4 embedding model.
- Phase 3 có đủ clustering runs cho KMeans, GMM, HDBSCAN và lưu bảng metrics.
- Phase 4 có propagation artifacts cho classification set.
- Phase 5 có code fine-tune classifier, Optuna, early stopping, metrics; `reports/results_table2.csv` đã tồn tại với 16 dòng kết quả.
- Phase 6 có reporting script và thư mục `reports/` chứa bảng, hình, report.

Tuy nhiên, nếu đánh giá theo tiêu chí "re-run local và tái lập hoàn toàn từ đầu đến cuối", repo hiện tại **chưa sạch hoàn toàn**. Các điểm cần note:

- `phase6_reporting.py` đang hard-code input ở `outputs/outputs/...`, nhưng workspace hiện tại chỉ có `outputs/phase3_clustering`, `outputs/phase4_cluster_propagation`, không có `outputs/outputs`.
- Không thấy thư mục local `outputs/phase5_classification` hoặc source file `results_table_all_runs.csv` tương ứng với Phase 5; chỉ thấy file report đã tổng hợp `reports/results_table2.csv`.
- `Makefile` và `run_smoke.ps1` đang dùng các flag như `--output-dir`, `--sample-size`, trong khi CLI code dùng `--output`, `--output_root`, `--phase1_output`, v.v. Các smoke targets vì vậy chưa phải reproducibility script chạy được ngay.
- Phase 6 hiện tại tạo `confusion_matrix_summary.csv` và `top_errors.csv` ở mức summary cấu hình, chưa phải confusion matrix thật và top document-level errors như plan yêu cầu.
- Phase 3 đang chọn HDBSCAN rất cao theo score trong một số bảng vì ARI/NMI được tính trên phần non-noise, trong khi noise fraction tới hơn 90%. Điều này làm HDBSCAN nhìn "rất tốt" trên một tập nhỏ dễ, nhưng không sát kết luận paper/plan rằng HDBSCAN kém do nhiều noise.

Nói ngắn gọn: **code các phase chính đã có và bám ý tưởng paper**, nhưng phần reporting/reproducibility cần chỉnh lại để artifact không bị stale và để kết luận thực nghiệm không bị lệch do đường dẫn, artifact ngoài repo, hoặc cách tính HDBSCAN.

---

## 2. Pipeline tổng thể

Paper đặt câu hỏi: nếu ta biến abstract khoa học thành các bộ ba tri thức `(subject, relation, object)` rồi đưa vào embedding/classification, liệu mô hình có hiểu tài liệu tốt hơn text-only hay không?

Pipeline repo hiện tại đi theo luồng:

```text
arXiv metadata JSONL
        |
        v
Phase 1: lọc dữ liệu, clean abstract, extract triples, tạo 4 representation
        |
        v
Phase 2: encode text thành embeddings bằng MiniLM, MPNet, SPECTER, SciBERT
        |
        v
Phase 3: clustering 5.000 docs bằng KMeans/GMM/HDBSCAN
        |
        v
Phase 4: propagate cluster signal sang 10.000 docs classification bằng nearest neighbors
        |
        v
Phase 5: fine-tune classifiers với text representation + cluster signal
        |
        v
Phase 6: tổng hợp Table 1, Table 2, hình, report, reproducibility notes
```

4 representation xuyên suốt pipeline:

- `abstract`: chỉ dùng abstract đã clean.
- `triples`: chỉ dùng các triples được linearize thành câu ngắn.
- `concatenate`: nối abstract và triples như một chuỗi phẳng.
- `hybrid`: nối abstract và triples bằng token phân tách `[SEP]`.

Vì sao cần 4 representation này? Vì paper muốn phân biệt 3 câu hỏi:

- Text gốc tự nó đã đủ tốt chưa?
- Triples riêng lẻ có giữ đủ ngữ nghĩa không?
- Khi kết hợp text + triples thì kết hợp kiểu nào hiệu quả hơn?

---

## 3. Phase 0 - Reproduction Contract

### Phase này nhằm làm gì?

Phase 0 không phải là bước xử lý ML, mà là "hợp đồng tái lập". Nó xác định trước dataset, seed, cấu trúc output, cách ghi config và các assumption để sau này kết quả không bị mơ hồ.

Trong reproduction, phase này rất quan trọng vì paper không công bố mọi chi tiết, đặc biệt là mốc "recent years" dùng để lọc arXiv. Nếu nhóm tự chọn mốc năm mà không ghi lại, kết quả sau này khó giải thích.

### Plan yêu cầu gì?

Theo `paper_reimplementation_todo.md`, Phase 0 cần:

- Dùng Kaggle/Cornell arXiv metadata snapshot.
- Ghi ngày snapshot, số dòng, dung lượng, checksum nếu có thể.
- Cố định seed `42`.
- Chuẩn hóa output ở `outputs/phase1_data/`, `outputs/phase2_embeddings/`, ...
- Ghi config chạy thí nghiệm vào JSON/YAML.
- Ghi rõ mốc năm nếu dùng filter vì paper không nói chính xác.

### Repo hiện tại đã làm tới đâu?

Đã có:

- Dataset local: `dataset/arxiv-metadata-oai-snapshot.json`, dung lượng khoảng 4.9 GB.
- Seed `42` xuất hiện trong các CLI và run config.
- Output chính hiện nằm đúng ở:
  - `outputs/phase1_data/`
  - `outputs/phase2_embeddings/`
  - `outputs/phase3_clustering/`
  - `outputs/phase4_cluster_propagation/`
  - `reports/`
- Các job Phase 2/3/4 có `run_config.json`, `phase2_job.json`, `propagation_config.json`, `propagation_summary.json`.
- Dữ liệu Phase 1 hiện có update year từ `2019` đến `2025`.

Chưa đầy đủ hoặc cần chỉnh:

- Chưa thấy checksum/số dòng snapshot được ghi thành artifact chính thức.
- CLI Phase 1 default vẫn là `output_triples`, trong khi plan/README muốn `outputs/phase1_data`.
- Một số `run_config.json` Phase 2 ghi `phase1_output: output_triples`, nhưng thư mục canonical hiện tại là `outputs/phase1_data`; điều này gây lệch provenance.
- `Makefile`/`run_smoke.ps1` không khớp CLI thật.

### Vì sao phải làm Phase 0?

Vì reproduction chỉ có ý nghĩa nếu người khác có thể trả lời:

- Dữ liệu nào được dùng?
- Có bao nhiêu record?
- Đã lọc theo năm nào?
- Seed là gì?
- Mỗi output sinh ra từ config nào?

Nếu thiếu Phase 0, cùng một code nhưng chạy snapshot khác, năm khác hoặc seed khác sẽ tạo kết quả khác, và nhóm sẽ khó biết sai khác là do mô hình hay do dữ liệu.

---

## 4. Phase 1 - Data Preparation, Triples, Knowledge Graph Edges

### Phase này làm gì?

Phase 1 biến arXiv metadata thô thành dữ liệu sạch cho embedding, clustering và classification.

Source chính:

- `data_processing/loading.py`
- `data_processing/triples.py`
- `data_processing/representations.py`
- `data_processing/output.py`
- `arxiv_triples_pipeline.py`

Output chính:

- `outputs/phase1_data/cluster_combined.jsonl`: 5.000 docs.
- `outputs/phase1_data/classify_combined.jsonl`: 10.000 docs.
- Các file theo representation: `cluster_abstract.jsonl`, `cluster_triples.jsonl`, `cluster_concatenate.jsonl`, `cluster_hybrid.jsonl` và tương tự cho `classify`.

### Các bước bên trong Phase 1

#### 4.1 Load và filter arXiv metadata

Code đọc JSONL streaming từ dataset arXiv. Mỗi record cần có:

- `abstract`
- `categories`
- `update_date`

Sau đó lấy `primary_category` là category đầu tiên, ví dụ `cs.AI`, và label top-level là phần trước dấu chấm, ví dụ `cs`.

Vì sao làm vậy?

- arXiv một paper có thể có nhiều category; dùng category đầu tiên là cách đơn giản để có một nhãn chính.
- Clustering/classification cần ground-truth label để tính ARI/NMI và supervised metrics.
- Top-level label giúp gom nhiều category con thành nhãn rộng hơn, ví dụ `cs.AI`, `cs.CL`, `cs.LG` đều thành `cs`.

Kết quả kiểm tra trong workspace:

- Cluster split có 5.000 docs.
- Classification split có 10.000 docs.
- Label phân bố không cân bằng, ví dụ cluster split có `cs` 1.763 docs, `math` 962 docs, `astro-ph` 404 docs, `cond-mat` 382 docs, ...
- Năm update hiện trải từ `2019` đến `2025`.

Điểm cần note:

- Label không chỉ là `cs.*`; dataset hiện có nhiều top-level archive như `math`, `astro-ph`, `cond-mat`, `physics`, `quant-ph`, ...
- Một số report Phase 6 ghi "Computer Science primary categories" là không chính xác với output hiện tại.

#### 4.2 Clean abstract

`clean_abstract()` làm:

- thay newline bằng space;
- xóa một số LaTeX pattern;
- xóa command LaTeX;
- lowercase;
- normalize whitespace.

Vì sao làm vậy?

- Transformer embedding và dependency parser dễ ổn định hơn khi text không có newline/LaTeX nhiễu.
- Lowercase giúp đồng nhất text, đặc biệt khi dùng model uncased như SciBERT uncased.
- Whitespace normalization làm representation reproducible hơn.

Điểm cần note:

- Code xóa LaTeX khá mạnh. Điều này có lợi cho text sạch, nhưng với math/physics abstract có thể làm mất ký hiệu quan trọng.
- Đây là một assumption implementation, vì paper chỉ mô tả preprocessing ở mức tổng quát.

#### 4.3 Split non-overlap

Code shuffle bằng seed `42`, rồi lấy:

- 5.000 docs đầu cho clustering.
- 10.000 docs tiếp theo cho classification.

Vì sao phải tách non-overlap?

- Clustering phase tạo cluster signal trên một tập riêng.
- Classification phase dùng tập khác để tránh việc classifier "nhìn lại" chính các document đã dùng để sinh cluster.
- Nếu hai tập overlap, propagation có thể trở thành gần như copy label và đánh giá bị optimistic.

Điểm cần note:

- Code hiện split bằng shuffle thường, không stratified theo label. Phase 6 report có chỗ ghi stratified là chưa khớp với Phase 1 code.
- Vì label imbalance khá rõ, non-stratified split có thể làm label hiếm xuất hiện rất ít.

#### 4.4 Extract triples bằng spaCy/scispaCy

`extract_triples()` duyệt từng sentence và từng token. Quy tắc:

- Relation anchor là token có POS `VERB` hoặc `AUX`.
- Subject lấy từ dependency `nsubj` hoặc `nsubjpass`.
- Object ưu tiên `dobj` hoặc `attr`.
- Nếu không có object trực tiếp, fallback qua `prep -> pobj/pcomp`.
- Mỗi triple giữ `source_sentence` và `rule_tag`.

Ví dụ output hiện tại:

```json
{
  "subj": "we",
  "rel": "consider",
  "obj": "a (nonlinear) transformation",
  "rule_tag": "dobj"
}
```

Vì sao paper cần triples?

- Abstract là text phi cấu trúc. Model phải tự suy ra quan hệ giữa concept.
- Triples ép text thành cấu trúc gần knowledge graph: ai/lý thuyết/phương pháp làm gì với đối tượng nào.
- Với tài liệu khoa học, quan hệ như "method improves performance", "model uses attention", "algorithm optimizes objective" có thể là tín hiệu phân loại hữu ích.

Vì sao phải lưu `source_sentence`?

- Triple extractor rule-based chắc chắn có nhiễu.
- Provenance giúp debug: nếu triple sai, ta nhìn lại câu gốc để biết parser sai ở subject, relation hay object.
- Đây cũng là nền cho extension Phase 7: chấm điểm chất lượng triple.

Kết quả kiểm tra:

- Cluster split trung bình khoảng `6.72` triples/doc.
- Classification split trung bình khoảng `6.81` triples/doc.
- Cluster split có 74 docs không extract được triple.
- Classification split có 145 docs không extract được triple.

#### 4.5 Tạo document-level KG edges

Mỗi triple được chuyển thành edge:

```text
source --relation--> target
```

Kèm:

- `source_sentence`
- `rule_tag`

Vì sao cần edge list?

- Với Phase 1-6 reproduction hiện tại, edge list chủ yếu là artifact để trace và kiểm chứng.
- Với Phase 7/8, edge list là dữ liệu gốc để xây graph features hoặc graph-aware representation.

#### 4.6 Tạo 4 representation

Code tạo:

- `fmt_abstract`: abstract sạch.
- `fmt_triples`: triples linearized dạng `Subject relation object.`
- `fmt_concatenate`: abstract + triples nối phẳng.
- `fmt_hybrid`: abstract + `[SEP]` + triples.

Vì sao cần linearize triples?

- Các model như MiniLM, MPNet, SciBERT, SPECTER nhận text sequence, không nhận graph trực tiếp.
- Linearization biến triples thành "câu" để transformer đọc được.

Vì sao `hybrid` dùng `[SEP]`?

- `[SEP]` báo cho transformer biết phần sau là một phân đoạn khác: text gốc và knowledge text không bị trộn lẫn hoàn toàn.
- Đây là điểm quan trọng trong paper: cùng là abstract + triples, cách phân tách có thể ảnh hưởng classification.

### Đánh giá Phase 1

Mức bám plan: **cao**.

Điểm tốt:

- Đúng 5.000/10.000 non-overlap.
- Có triple extraction theo dependency rule.
- Có provenance và KG edges.
- Có đủ 4 representation.
- Output JSONL/CSV rõ ràng.

Điểm cần chỉnh:

- Default output CLI nên đổi về `outputs/phase1_data`.
- Ghi snapshot metadata/checksum vào artifact.
- Báo cáo lại đúng label scope: không chỉ Computer Science.
- Cần thêm script inspect 10 triples hoặc ghi sample triples vào report chính.

---

## 5. Phase 2 - Embedding Generation

### Phase này làm gì?

Phase 2 biến text representation thành vector embedding để dùng cho clustering và nearest-neighbor propagation.

Source chính:

- `embeddings/embedding_pipeline.py`
- `embeddings/encoders.py`
- `embeddings/io_utils.py`
- `embeddings/cli.py`

Output:

```text
outputs/phase2_embeddings/{split}/{representation}/{model_slug}/
  embeddings.npy
  metadata.jsonl
  verification.json
  run_config.json
```

Workspace hiện có 32 file `embeddings.npy`, đúng với:

```text
2 splits x 4 representations x 4 models = 32 embedding jobs
```

### Models được dùng

Phase 2 support:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `allenai/specter`
- `allenai/scibert_scivocab_uncased`

Vì sao chọn các model này?

- MiniLM: nhẹ, nhanh, baseline sentence-transformer.
- MPNet: sentence-transformer mạnh hơn, thường tốt cho semantic similarity.
- SPECTER: embedding scientific document được học bằng citation signals.
- SciBERT: BERT pretrain trên scientific text, hợp domain khoa học.

Paper so sánh các model này để trả lời: model general-purpose nhẹ có đủ tốt không, hay model domain-specific mới tốt hơn?

### Pooling và normalization

Với SentenceTransformer:

- dùng `model.encode()`;
- có thể normalize trực tiếp.

Với SPECTER/SciBERT qua Transformers:

- tokenize với `max_length`;
- lấy hidden states;
- pooling `mean` hoặc `cls`, default `mean`;
- L2-normalize vector.

Vì sao cần pooling?

- Transformer trả embedding cho từng token, nhưng clustering cần một vector/document.
- Mean pooling gom toàn bộ token thành document representation.

Vì sao cần L2-normalize?

- Embedding magnitude khác nhau có thể làm KMeans/GMM/nearest-neighbor bị bias.
- Sau normalize, cosine/euclidean trên vector đơn vị ổn định hơn.
- Verification hiện xác nhận norm gần 1, ví dụ sample có norm min/max xấp xỉ `1.0`.

### Metadata mapping

Mỗi embedding job lưu `metadata.jsonl` gồm:

- `row_index`
- `id`
- `label`
- `primary_category`
- `n_triples`
- `text_num_chars`

Vì sao phải có metadata?

- `embeddings.npy` chỉ là matrix, không biết dòng nào là document nào.
- Clustering cần `label` để tính ARI/NMI.
- Propagation cần đảm bảo thứ tự embedding khớp với doc IDs.

### Đánh giá Phase 2

Mức bám plan: **cao**.

Điểm tốt:

- Có đủ 4 models x 4 representations x 2 splits.
- Có `.npy`, metadata, verification, run config.
- Có check NaN/Inf/zero norm/unit norm.
- Có pooling thống nhất cho transformer models.

Điểm cần chỉnh:

- `embeddings/README.md` dùng command `python -m phase2_embeddings`, nhưng package hiện tại là `embeddings`; cần sửa doc.
- Một số `run_config.json` ghi `phase1_output: output_triples` trong khi canonical output hiện là `outputs/phase1_data`.
- Nếu muốn đúng plan tuyệt đối, cần ghi rõ embedding dimension thực tế trong report cuối, không chỉ trong run config.

---

## 6. Phase 3 - Unsupervised Clustering

### Phase này làm gì?

Phase 3 chạy clustering trên 5.000 document clustering split.

Source chính:

- `clustering/clustering_pipeline.py`
- `clustering/io_utils.py`
- `clustering/cli.py`

Input:

- embeddings từ `outputs/phase2_embeddings/cluster/...`

Output chính:

- `outputs/phase3_clustering/results_table.csv`
- `outputs/phase3_clustering/results_table_best_by_algorithm.csv`
- `outputs/phase3_clustering/results_table_best_by_representation.csv`
- cluster labels `.npy`
- cluster analysis CSV/JSON cho best configs

Workspace hiện có:

- `416` experiment rows trong `outputs/phase3_clustering/results_table.csv`.
- Breakdown:
  - KMeans: 160 rows.
  - GMM: 160 rows.
  - HDBSCAN: 96 rows.

### Thuật toán clustering

Phase 3 chạy:

- KMeans với `k = 3..12`.
- Gaussian Mixture Model với `k = 3..12`.
- HDBSCAN với `min_cluster_size = 5, 10, 20, 30, 50, 100`.

Vì sao dùng nhiều thuật toán?

- KMeans giả định cụm gần spherical và cần `k`.
- GMM mềm hơn KMeans, cho phép cụm gaussian với covariance.
- HDBSCAN không cần định trước `k`, có thể phát hiện noise.

Paper dùng cả ba để xem document embedding space có tự tách theo lĩnh vực không.

### Metrics

Phase 3 tính:

- ARI: đo độ khớp giữa cluster assignment và ground-truth label, đã điều chỉnh ngẫu nhiên.
- NMI: đo lượng thông tin chung giữa cluster labels và true labels.
- Silhouette: đo độ tách cụm trong embedding space.
- Noise fraction: riêng cho HDBSCAN.

Vì sao không chỉ dùng accuracy?

- Clustering không biết mapping cluster -> class.
- ARI/NMI phù hợp vì chúng không cần cluster id trùng label id.
- Silhouette không cần true label, giúp xem structure nội tại của embedding space.

### Cách chọn best

Code dùng:

- KMeans/GMM score = `0.5 * ARI + 0.5 * NMI`.
- HDBSCAN score = `NMI + 0.5 * ARI - 0.5 * noise_fraction`.

Vì sao cần score tổng hợp?

- ARI và NMI nhìn hai khía cạnh khác nhau.
- Một config có ARI tốt nhưng NMI thấp có thể không ổn định.
- Score giúp chọn config cho Phase 4.

### Kết quả hiện tại và caveat HDBSCAN

Khi nhìn toàn bộ score, HDBSCAN đang đứng rất cao. Ví dụ:

- `abstract + MiniLM + HDBSCAN(min_cluster_size=50)` có ARI khoảng `0.9417`, NMI khoảng `0.8993`, nhưng noise fraction khoảng `0.9244`.
- Tức là hơn 92% document bị gán noise, chỉ phần non-noise nhỏ được đánh giá ARI/NMI rất cao.

Điều này cần cẩn trọng:

- Nếu ARI/NMI tính trên non-noise only, HDBSCAN có thể "ăn điểm" bằng cách chỉ cluster một số điểm dễ.
- Paper/plan kỳ vọng HDBSCAN kém vì nhiều noise.
- Do đó khi viết báo cáo Table 1, nên tách riêng:
  - best KMeans/GMM để so sánh sát paper;
  - HDBSCAN kèm noise fraction để không hiểu nhầm.

Nếu bỏ HDBSCAN, best non-HDBSCAN hiện tại là:

- `abstract + SciBERT + GMM(k=5)`: ARI khoảng `0.5305`, NMI khoảng `0.5277`.
- Một số config `abstract/hybrid + SPECTER` với KMeans/GMM k=4 cũng quanh ARI `0.49`.

So với expected trong plan:

- Paper/plan kỳ vọng Abstract + MPNet KMeans/GMM tốt nhất, ARI khoảng `0.47`, NMI khoảng `0.55`.
- Repo hiện tại non-HDBSCAN best ARI cao hơn một chút (`0.53`) nhưng NMI thấp hơn target (`0.528` so với `0.55`), và model best khác (`SciBERT` thay vì MPNet).

### Cluster analysis

Code có phân tích:

- cluster assignments;
- label distribution trong từng cluster;
- cluster purity;
- weighted purity;
- largest cluster fraction;
- noise fraction.

Vì sao cần phân tích label distribution?

- ARI/NMI là số tổng hợp, khó giải thích.
- Label distribution cho biết cluster có thật sự tương ứng domain không.
- Paper phân tích các cluster như `astro-ph`, `math`, `cond-mat vs quant-ph`, `hep/ph/gr-qc`, interdisciplinary clusters; repo có nền dữ liệu để làm các phân tích này.

### Đánh giá Phase 3

Mức bám plan: **khá cao về implementation, cần chỉnh cách diễn giải HDBSCAN**.

Điểm tốt:

- Đủ KMeans/GMM/HDBSCAN.
- Đủ metrics.
- Có labels `.npy`.
- Có cluster purity/label distribution.

Điểm cần chỉnh:

- `results_table_best_by_representation.csv` hiện có thể chọn HDBSCAN với noise >90%; không nên dùng bảng này trực tiếp để kết luận "best representation".
- Nên tạo thêm bảng `best_by_representation_non_hdbscan.csv` hoặc tính HDBSCAN ARI/NMI trên toàn bộ docs, coi `-1` là một cluster.
- Phase 6 summary nói 900+ rows là không khớp output hiện tại 416 rows.

---

## 7. Phase 4 - Cluster Propagation sang Classification Set

### Phase này làm gì?

Phase 4 lấy cluster labels học từ 5.000 docs clustering và gán một cluster signal cho 10.000 docs classification.

Source chính:

- `propagation/cluster_propagation_pipeline.py`
- `propagation/io_utils.py`
- `propagation/cli.py`

Input:

- cluster embeddings và labels từ Phase 2/3;
- classify embeddings từ Phase 2.

Output:

```text
outputs/phase4_cluster_propagation/{representation}/{model_slug}/{algorithm_config}/
  propagated_clusters.jsonl
  propagated_clusters.csv
  propagation_config.json
  propagation_summary.json
```

Workspace hiện có 32 `propagation_summary.json`, tương ứng:

```text
4 representations x 4 models x 2 algorithms(KMeans/GMM) = 32 jobs
```

### Cách propagation hoạt động

Với mỗi classification document:

1. Lấy embedding của document đó trong cùng representation/model.
2. Tìm `neighbor_k` nearest neighbors trong 5.000 clustering docs.
3. Lấy cluster labels của các neighbors.
4. Gán propagated cluster bằng majority vote hoặc distance-weighted vote.
5. Lưu thêm confidence, distances, neighbor ids, neighbor labels.

Default hiện tại:

- `neighbor_k = 5`
- metric `cosine`
- assignment rule `majority_vote`
- handle HDBSCAN noise: `ignore`

Ví dụ summary:

- `abstract + SciBERT + GMM(k=5)` propagate 10.000 docs vào 5 cluster.
- Largest propagated cluster fraction khoảng `0.4744`.
- Mean confidence khoảng `0.9298`.

### Vì sao cần Phase 4?

Paper có nhắc việc dùng clustering signal cho classification nhưng không mô tả chi tiết toàn bộ implementation. Phase 4 là cầu nối giữa unsupervised và supervised:

- Clustering phát hiện cấu trúc latent trong document space.
- Propagation biến cấu trúc đó thành feature cho classifier.
- Classifier không chỉ thấy text, mà còn biết document này gần cụm nào trong embedding space.

### Vì sao dùng nearest neighbors?

Vì clustering labels chỉ tồn tại trên 5.000 docs clustering. Với 10.000 docs classification, ta cần một cách gán cluster id mà không re-cluster chung hai tập. Nearest neighbor là cách đơn giản, minh bạch:

- Nếu classification doc gần các cluster docs cùng cluster, ta gán cluster đó.
- Confidence cho biết neighbor vote có nhất quán không.
- Không cần train thêm model.

### Đánh giá Phase 4

Mức bám plan: **cao, nhưng đây là assumption implementation**.

Điểm tốt:

- Có propagation riêng theo representation/model.
- Có kiểm tra alignment doc ids.
- Có summary distribution để xem cluster có collapse không.
- Không propagate HDBSCAN mặc định, tránh noise quá cao.

Điểm cần chỉnh:

- Paper không mô tả chi tiết propagation, nên report phải ghi rõ đây là cách nhóm hiện thực.
- Nên thêm bảng tổng hợp propagation collapse/noise cho mọi representation/model vào report cuối.
- Nếu Phase 5 chỉ dùng một best propagation per representation/model, cần ghi rõ chọn theo Phase 3 score nào.

---

## 8. Phase 5 - Supervised Classification

### Phase này làm gì?

Phase 5 fine-tune Transformer classifiers trên 10.000 docs classification.

Source chính:

- `classification/classification_pipeline.py`
- `classification/training.py`
- `classification/metrics.py`
- `classification/io_utils.py`
- `classification/cli.py`

Mục tiêu là tái hiện Table 2 của paper: so sánh các cặp:

```text
clustering representation x classifier input representation
```

### Input classifier thực tế

Với mỗi example, code lấy:

- text base theo `classifier_representation`;
- propagated cluster id/confidence từ Phase 4 theo `clustering_representation`;
- inject cluster signal vào text theo `cluster_signal_mode`.

Default:

```text
clustering mode {rep}. propagated cluster {id}. confidence {score}. [SEP] {base_text}
```

Vì sao làm vậy?

- Classifier Transformer nhận text, không nhận numeric cluster id trực tiếp.
- Cluster signal được verbalize thành text prefix để classifier dùng cùng tokenizer/model.
- `[SEP]` tách cluster signal với nội dung document.

Điểm rất quan trọng:

- Đây là một implementation choice. Nếu paper Table 2 chỉ fine-tune trên text representation mà không verbalize cluster signal theo kiểu này, kết quả repo sẽ không hoàn toàn tương đương.
- Nếu muốn so sánh "text representation only", có thể chạy `--cluster_signal_mode none`.

### Training protocol

Code hiện:

- dùng `AutoModelForSequenceClassification`;
- tokenizer max length default `128`;
- stratified train/validation split `80/20`;
- Optuna search:
  - learning rate `[1e-6, 1e-4]`;
  - batch size `{8, 16, 32}`;
  - epochs `2..7`;
- AdamW + linear warmup schedule;
- early stopping theo validation loss;
- seed control cho Python/NumPy/Torch.

Vì sao max length 128?

- Abstract/triples có thể dài, nhưng fine-tune Transformer tốn GPU.
- 128 token là trade-off để chạy nhiều experiment.
- Paper/plan cũng đặt max length `128`.

Vì sao dùng Optuna?

- Learning rate, batch size, epochs ảnh hưởng mạnh tới fine-tuning.
- Optuna giúp tìm config tốt hơn thay vì chọn tay một bộ cố định.

### Metrics

`classification/metrics.py` tính:

- accuracy;
- macro precision/recall/F1;
- weighted precision/recall/F1;
- Cohen's kappa;
- MCC;
- top-3 accuracy;
- macro ROC-AUC one-vs-rest nếu tính được.

Vì sao dùng nhiều metrics?

- Dataset label imbalance; accuracy có thể cao do class lớn như `cs`, `math`.
- Macro-F1 xem class hiếm có được học không.
- Weighted-F1 phản ánh phân bố thật.
- MCC và Cohen's kappa tốt hơn accuracy khi imbalance.
- Top-3 accuracy cho biết model có đưa true label vào shortlist không.

### Experiment plan hiện tại

Code có hai mode:

- `table2`: chạy 16 dòng paper-reported, mỗi pair chọn model alias định trước trong `PAPER_TABLE2_EXPERIMENTS`.
- `full_grid`: chạy toàn bộ cartesian product model x clustering rep x classifier rep.

Điểm cần note:

- Plan Phase 5 nói "Models cần chạy theo paper: SciBERT, SPECTER, MiniLM" và "16 experiment chính theo Table 2". Nếu hiểu là 16 pairs x 3 models = 48 runs, thì default `table2` chưa chạy full 48; cần dùng `full_grid`.
- CLI default `optuna_trials=3`, trong khi Phase 6 report có chỗ ghi "10-trial sweep"; cần chỉnh cho nhất quán.

### Kết quả hiện có

Trong workspace hiện tại:

- Không thấy thư mục local `outputs/phase5_classification`.
- Không thấy source `results_table_all_runs.csv` trong `outputs/...`.
- Có `reports/results_table2.csv` với 16 kết quả đã tổng hợp.

Best trong `reports/results_table2.csv`:

- Clustering mode: `abstract`
- Classifier input: `hybrid`
- Model: `scibert`
- Accuracy: `0.8140`
- Macro-F1: khoảng `0.5314`
- Top-3 accuracy: `0.9630`

So với expected paper/plan:

- Paper/plan target: accuracy khoảng `0.926`, macro-F1 khoảng `0.925`.
- Repo report hiện thấp hơn rất nhiều về macro-F1.

Diễn giải hợp lý:

- `macro-F1` thấp trong khi `weighted-F1`/accuracy cao cho thấy model học tốt class lớn nhưng yếu ở class hiếm.
- Label top-level trong output hiện rất imbalance và có nhiều label hiếm chỉ vài chục docs.
- Phase 5 có thể đã chạy ngoài workspace/Kaggle, nên thiếu artifact để kiểm chứng split, predictions, confusion matrix thật.
- Nếu chỉ có 16 runs và Optuna 3 trials, search có thể chưa đủ sâu.

### Đánh giá Phase 5

Mức bám plan: **code bám tốt, artifact local chưa đủ để gọi là reproducible hoàn toàn**.

Điểm tốt:

- Fine-tune đúng `AutoModelForSequenceClassification`.
- Có stratified 80/20.
- Có Optuna + early stopping.
- Có metrics đầy đủ.
- Có output predictions/confusion/top-errors ở cấp trial nếu chạy local.

Điểm cần chỉnh:

- Đưa source Phase 5 outputs vào repo/output canonical hoặc chỉnh Phase 6 đọc đúng nơi.
- Ghi rõ `cluster_signal_mode` vì đây là khác biệt implementation.
- Nếu muốn claim 16 x 3 models, chạy `full_grid`.
- Tạo document-level `top_errors.csv` từ predictions thật, không chỉ worst configs.

---

## 9. Phase 6 - Reporting and Reproducibility

### Phase này làm gì?

Phase 6 tổng hợp artifacts từ Phase 1-5 thành report:

- Table 1 clustering.
- Table 2 classification.
- Confusion/error analysis.
- Cluster analysis summary.
- Figures.
- Reproduction report.
- Smoke scripts/Makefile.

Source chính:

- `phase6_reporting.py`
- `Phase6_Reporting_Reproducibility_Fixed.ipynb`
- `PHASE6_README.md`
- `PHASE6_EXECUTION_GUIDE.md`
- `PHASE6_IMPLEMENTATION_SUMMARY.md`

Generated artifacts hiện có trong `reports/`:

- `results_table1.csv`: 416 rows.
- `results_table1_summary.csv`.
- `results_table1.md`.
- `results_table2.csv`: 16 rows.
- `results_table2_best_by_pair.csv`.
- `results_table2.md`.
- `confusion_matrix_summary.csv`.
- `top_errors.csv`.
- `cluster_analysis_summary.json`.
- `environment_summary.json`.
- `reproduction_report.md`.
- 4 figures trong `reports/figures/`.

### Vì sao cần Phase 6?

Các phase trước sinh rất nhiều file phân tán. Phase 6 biến chúng thành câu chuyện khoa học:

- Setup dữ liệu là gì?
- Triple extraction hoạt động ra sao?
- Clustering model nào tốt?
- Classification model nào tốt?
- Kết quả lệch paper ở đâu?
- Có thể tái lập bằng command nào?

Không có Phase 6, repo chỉ là một đống script và CSV. Có Phase 6, nhóm có thể viết báo cáo, slide, hoặc supplementary material.

### Đánh giá Phase 6 hiện tại

Mức bám plan: **trung bình; artifact có, nhưng tính reproducibility còn yếu**.

Điểm tốt:

- Có file report tổng hợp.
- Có tables và figures.
- Có environment summary.
- Có documentation hướng dẫn.

Điểm chưa sát plan:

- `phase6_reporting.py` hard-code:

```python
PHASE3_DIR = Path("outputs/outputs/phase3_clustering")
PHASE4_DIR = Path("outputs/outputs/phase4_cluster_propagation")
PHASE5_DIR = Path("outputs/outputs/da-ktdl-phase5-table2")
```

Trong workspace hiện tại không có `outputs/outputs`. Vì vậy nếu re-run script ngay, nó không đọc được Phase 3/5 source hiện tại.

- `create_confusion_matrix_and_errors()` không đọc predictions thật; nó chỉ tạo summary từ Table 2.
- `create_top_errors_summary()` lấy worst configurations, không lấy top document-level errors gồm abstract/triples/true/predicted như plan.
- `create_cluster_analysis()` chỉ ghi placeholder, trong khi Phase 3 thật đã có cluster purity files.
- `create_smoke_test_script()` và `create_makefile()` sinh command dùng flag không tồn tại.
- `reproduction_report.md` có vài thông tin lệch:
  - nói dataset là Computer Science primary categories, trong khi output có nhiều top-level arXiv labels;
  - nói split stratified ở Phase 1, nhưng Phase 1 code shuffle thường;
  - nói Optuna 10 trials, CLI default là 3;
  - command dùng `--output-dir`, trong khi code dùng tên flag khác.

### Nên hiểu Phase 6 hiện tại như thế nào?

Phase 6 hiện tại hữu ích như một bản report tổng hợp đã generate, nhưng chưa nên dùng như bằng chứng "chạy lại từ đầu chắc chắn ra đúng vậy" nếu chưa sửa path và artifacts.

Nên xem nó là:

- good first reporting draft;
- cần hardening trước khi nộp/báo cáo chính thức.

---

## 10. So sánh bám plan từng phase

| Phase | Mục tiêu chính | Trạng thái hiện tại | Đánh giá |
|---|---|---|---|
| Phase 0 | Reproduction contract, seed, output, config | Có seed/output/config rải rác, thiếu checksum và có mismatch path | Cần chỉnh |
| Phase 1 | Load/filter/split/extract triples/4 reps | Có code và output 5k/10k, bám rất sát | Tốt |
| Phase 2 | Generate embeddings 4 models x 4 reps | Có 32 embedding jobs, verification OK | Tốt |
| Phase 3 | Clustering KMeans/GMM/HDBSCAN + metrics | Có 416 runs và cluster analysis | Tốt nhưng cần xử lý HDBSCAN noise khi kết luận |
| Phase 4 | Propagate cluster signal | Có 32 propagation jobs KMeans/GMM | Tốt, nhưng phải ghi rõ assumption |
| Phase 5 | Fine-tune classification | Code đầy đủ, report có kết quả, nhưng artifact local Phase 5 thiếu | Chưa hoàn toàn reproducible |
| Phase 6 | Report/reproducibility | Reports tồn tại, script path/smoke/confusion/top-errors chưa sát | Cần chỉnh trước khi claim hoàn tất |

---

## 11. Các điểm lệch quan trọng so với paper/plan

### 11.1 HDBSCAN đang bị diễn giải quá tốt

HDBSCAN có thể đạt ARI/NMI rất cao vì metrics được tính trên non-noise subset. Nhưng nếu hơn 90% docs là noise, kết luận "HDBSCAN tốt" là sai tinh thần paper.

Khuyến nghị:

- Trong report chính, không chọn HDBSCAN làm best overall nếu noise fraction quá cao.
- Thêm điều kiện chọn best HDBSCAN, ví dụ `noise_fraction < 0.5`.
- Hoặc tính ARI/NMI trên toàn bộ docs, coi noise `-1` như một cluster.

### 11.2 Phase 5 result thấp hơn paper rất nhiều

Best report hiện tại:

- Accuracy `0.8140`.
- Macro-F1 `0.5314`.

Expected plan:

- Accuracy `0.926`.
- Macro-F1 `0.925`.

Chênh lệch lớn cần được giải thích trước khi báo cáo:

- label imbalance;
- khác dataset snapshot/year filter;
- khác split;
- Optuna trials ít;
- artifact Phase 5 chạy ngoài workspace;
- cluster signal injection khác implementation paper;
- có thể classification đang dùng top-level labels nhiều domain hiếm hơn paper.

### 11.3 Phase 6 không đọc đúng input hiện tại

Script đang trỏ `outputs/outputs/...`, nhưng outputs hiện là `outputs/phase3_clustering`, `outputs/phase4_cluster_propagation`. Điều này làm report có nguy cơ stale.

Khuyến nghị:

- Đổi constants hoặc thêm CLI args:
  - `--phase3-dir outputs/phase3_clustering`
  - `--phase4-dir outputs/phase4_cluster_propagation`
  - `--phase5-dir outputs/phase5_classification`
- Nếu Phase 5 kết quả đến từ Kaggle, copy đầy đủ source artifacts vào `outputs/phase5_classification/`.

### 11.4 Smoke scripts chưa chạy được

Các generated commands dùng `--output-dir` và `--sample-size`, nhưng CLI không nhận các flag này.

Khuyến nghị sửa:

```bash
python -m arxiv_triples_pipeline --output outputs/phase1_smoke --n_cluster 100 --n_classify 200
python -m embeddings --phase1_output outputs/phase1_smoke --output_root outputs/phase2_smoke ...
python -m clustering --phase2_root outputs/phase2_smoke --output_root outputs/phase3_smoke ...
```

### 11.5 Confusion matrix và top errors chưa đúng yêu cầu plan

Plan muốn:

- confusion matrix cho best classifier;
- top errors gồm abstract, triples, true label, predicted label.

Phase 6 hiện tại chỉ tạo summary từ Table 2.

Khuyến nghị:

- Đọc `validation_predictions.csv` và `confusion_matrix.csv` từ best trial của Phase 5.
- Join predictions với `classify_combined.jsonl` để lấy abstract/triples.
- Lưu `reports/top_document_errors.csv`.

---

## 12. Vì sao pipeline này có ý nghĩa khoa học?

Pipeline này không chỉ là chạy model. Nó kiểm tra một giả thuyết:

> Scientific document classification/clustering có thể hưởng lợi khi ta kết hợp text phi cấu trúc với tri thức có cấu trúc dạng triples.

Mỗi phase phục vụ giả thuyết này:

- Phase 1 tạo tri thức có cấu trúc từ abstract.
- Phase 2 đưa text/triples vào cùng không gian vector để so sánh công bằng.
- Phase 3 kiểm tra liệu representation có tự hình thành cụm theo domain không.
- Phase 4 biến structure unsupervised thành signal cho supervised task.
- Phase 5 kiểm tra signal đó có cải thiện classification không.
- Phase 6 biến kết quả thành bằng chứng có thể đọc, so sánh, tái lập.

Điểm hay của approach:

- Modular: có thể thay triple extractor, embedding model, clustering algorithm hoặc classifier mà không phá toàn pipeline.
- Explainable hơn text-only: triples có provenance sentence.
- Mở đường cho extension: triple quality filtering, graph features, graph neural models.

Điểm yếu tự nhiên:

- Rule-based triples nhiễu.
- Triple-only mất ngữ cảnh nên thường thấp hơn abstract.
- Scientific abstracts nhiều LaTeX và thuật ngữ chuyên ngành, parser có thể sai.
- Label arXiv top-level không luôn phản ánh topic thật của bài.
- Cluster propagation là assumption, cần báo cáo minh bạch.

---

## 13. Checklist cần làm trước khi tuyên bố "Phase 6 hoàn tất hoàn toàn"

Ưu tiên cao:

- Sửa `phase6_reporting.py` để đọc đúng `outputs/phase3_clustering`, `outputs/phase4_cluster_propagation`, `outputs/phase5_classification`.
- Copy/commit Phase 5 raw artifacts vào workspace hoặc ghi rõ chúng đến từ Kaggle và kèm link/export.
- Sửa `Makefile` và `run_smoke.ps1` theo CLI thật.
- Sinh confusion matrix thật và top document-level errors thật.
- Tách HDBSCAN khỏi best overall hoặc thêm noise-threshold.

Ưu tiên trung bình:

- Ghi dataset checksum/snapshot metadata.
- Thêm report về label distribution và imbalance.
- Thêm sample triples đúng/sai có provenance.
- Thêm bảng best non-HDBSCAN cho Table 1.
- Chạy `classification --experiment_plan full_grid` nếu muốn claim 48 runs.

Ưu tiên thấp nhưng tốt cho báo cáo:

- Vẽ distribution số triples/doc.
- Vẽ noise fraction HDBSCAN theo representation/model.
- Báo cáo per-label F1 để giải thích macro-F1 thấp.
- Ghi rõ paper local là v1 và repo follow v1.

---

## 14. Tóm tắt cho người đọc mới

Nếu bạn mới đọc repo này, hãy hiểu như sau:

1. Dataset arXiv được lọc thành 15.000 papers, gồm 5.000 để clustering và 10.000 để classification.
2. Mỗi abstract được clean và parse để trích các triples `(subject, relation, object)`.
3. Từ abstract và triples, repo tạo 4 kiểu text input: abstract, triples-only, concatenate, hybrid `[SEP]`.
4. Mỗi kiểu input được encode bằng 4 embedding models.
5. Embeddings của 5.000 docs được clustering bằng KMeans/GMM/HDBSCAN để xem representation nào tạo cụm topic tốt.
6. Cluster labels được propagate sang 10.000 docs classification bằng nearest neighbors.
7. Classifier Transformer được fine-tune trên text input có thể kèm cluster signal.
8. Report cuối tổng hợp tables/figures, nhưng phần reproducibility hiện cần sửa thêm để mọi thứ chạy lại local đúng như artifact.

Kết luận kỹ thuật: **pipeline clone lại đúng khung phương pháp của paper, nhưng chưa nên coi là reproduction hoàn hảo cho tới khi sửa Phase 6 paths, artifact Phase 5, smoke scripts, và cách diễn giải HDBSCAN.**

---

## 15. Tài liệu/thư mục nên đọc tiếp

- `paper_reimplementation_todo.md`: plan và expected targets.
- `summary_paper.md`: tóm tắt paper dễ đọc.
- `data_processing/triples.py`: logic extract triples quan trọng nhất.
- `embeddings/encoders.py`: cách encode và pooling.
- `clustering/clustering_pipeline.py`: metrics và cách chọn best cluster.
- `propagation/cluster_propagation_pipeline.py`: nearest-neighbor propagation.
- `classification/classification_pipeline.py`: cách build experiment Table 2.
- `classification/training.py`: fine-tune, Optuna, early stopping, metrics artifacts.
- `phase6_reporting.py`: report generator, cũng là file cần sửa nhiều nhất để reproducibility sạch.
