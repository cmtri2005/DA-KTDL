# GAP Implementation Notes

File này chuẩn bị câu trả lời chi tiết cho 3 nhóm câu hỏi quan trọng khi báo cáo:

1. Vì sao kết quả classification của repo thấp hơn paper?
2. Vì sao không nên claim HDBSCAN là tốt nhất dù ARI/NMI rất cao?
3. Vì sao Phase 4 cluster propagation là assumption của nhóm và phải trình bày thế nào?

Mục tiêu là giúp nhóm trả lời thầy theo hướng trung thực, kỹ thuật, và vẫn bảo vệ được công sức implementation.

---

## 1. Classification thấp hơn paper: giải thích thế nào?

### Câu hỏi thầy có thể hỏi

"Paper báo accuracy khoảng 92.6%, macro-F1 khoảng 0.925. Tại sao kết quả nhóm chỉ khoảng 81.4% accuracy và macro-F1 khoảng 0.53? Có phải implementation sai không?"

### Câu trả lời ngắn nên nói

Không nên trả lời là "paper sai" hay "model chạy chưa tốt" một cách chung chung. Cách nói an toàn:

> "Nhóm em đã clone lại đầy đủ pipeline và có chạy ra kết quả classification, nhưng kết quả hiện tại thấp hơn paper. Nguyên nhân chính nhiều khả năng đến từ khác biệt dataset/filtering, imbalance label rất mạnh, artifact Phase 5 chưa được chuẩn hóa lại vào workspace, và một số assumption implementation như cách inject cluster signal. Vì vậy nhóm em xem đây là kết quả reproduction có gap, không claim là đã match tuyệt đối paper."

### Kết quả hiện tại trong repo

Trong `reports/results_table2.csv`, best row hiện là:

```text
Clustering mode : abstract
Classifier input: hybrid
Model           : scibert
Accuracy        : 0.8140
Macro-F1        : 0.5314
Weighted-F1     : 0.8089
MCC             : 0.7698
Cohen Kappa     : 0.7696
Top-3 Accuracy  : 0.9630
```

So với target trong plan/paper summary:

```text
Paper expected accuracy : ~0.926
Paper expected macro-F1 : ~0.925
Repo current accuracy   : 0.8140
Repo current macro-F1   : 0.5314
```

Khoảng cách:

- Accuracy thấp hơn khoảng 11.2 điểm phần trăm.
- Macro-F1 thấp hơn rất nhiều, khoảng 39.4 điểm phần trăm.

Điểm đáng chú ý là `accuracy` và `weighted-F1` quanh 0.81, nhưng `macro-F1` chỉ quanh 0.53. Điều này thường có nghĩa là model học tốt các lớp lớn, nhưng yếu ở các lớp nhỏ/hiếm.

### Bằng chứng về imbalance label

Classification split hiện có 10.000 docs và 21 top-level labels:

```text
cs        : 3590
math      : 1900
cond-mat  : 804
physics   : 750
astro-ph  : 739
eess      : 430
quant-ph  : 405
stat      : 276
hep-ph    : 261
hep-th    : 188
gr-qc     : 168
q-bio     : 108
math-ph   : 71
econ      : 57
nucl-th   : 57
hep-ex    : 53
nlin      : 46
q-fin     : 40
nucl-ex   : 30
hep-lat   : 26
alg-geom  : 1
```

Hai lớp lớn nhất `cs` và `math` đã chiếm khoảng `54.9%` toàn bộ classification split. Có 9 labels dưới 100 documents, thậm chí `alg-geom` chỉ có 1 document.

Điều này ảnh hưởng trực tiếp tới macro-F1:

- Accuracy bị chi phối bởi lớp lớn.
- Weighted-F1 cũng bị chi phối bởi lớp lớn.
- Macro-F1 tính trung bình đều trên mọi label, nên label hiếm làm điểm tụt mạnh.

Nếu paper sử dụng filter khác, số label khác, hoặc loại các class quá hiếm, macro-F1 có thể cao hơn nhiều.

### Các nguyên nhân gap có khả năng cao

#### 1. Dataset filtering không chắc giống paper

Paper nói dùng subset "recent years" nhưng không công bố chính xác mốc lọc. Repo hiện tại dùng dữ liệu có `update_date` từ `2019` đến `2025`.

Nếu paper lọc theo:

- năm khác;
- category khác;
- chỉ một nhóm ngành nhất định;
- loại labels quá hiếm;
- hoặc lấy sample cân bằng hơn;

thì kết quả classification sẽ khác nhiều.

Cách trả lời:

> "Do paper không công bố chính xác rule lọc recent years, nhóm em phải chọn một reproduction contract riêng. Đây là một nguồn sai khác chính và nhóm em đã ghi nhận là assumption."

#### 2. Label space hiện tại rộng và imbalance

Repo map label từ category đầu tiên sang top-level archive, ví dụ:

```text
cs.AI -> cs
math.AG -> math
astro-ph.IM -> astro-ph
```

Việc này tạo 21 labels, trong đó nhiều labels rất ít mẫu. Với train/validation split 80/20, một label có 26 docs chỉ còn khoảng 5 docs validation; label có 1 doc gần như không học/evaluate ổn định được.

Cách trả lời:

> "Accuracy vẫn tương đối cao nhưng macro-F1 thấp vì macro-F1 phạt nặng các lớp hiếm. Đây là tín hiệu cho thấy dataset split hiện tại mất cân bằng hơn setup paper hoặc cần xử lý imbalance tốt hơn."

#### 3. Phase 5 artifacts chưa nằm đầy đủ trong workspace

Trong workspace hiện tại:

- Có `reports/results_table2.csv`.
- Nhưng không thấy local folder `outputs/phase5_classification/`.
- Không thấy source `results_table_all_runs.csv` trong `outputs/...` như `phase6_reporting.py` đang kỳ vọng.

Điều này nghĩa là Phase 5 có thể đã được chạy/export từ môi trường khác như Kaggle, nhưng raw artifacts chưa được đưa đầy đủ về repo.

Cách trả lời:

> "Phần code Phase 5 đã có đủ fine-tuning, Optuna, metrics, nhưng nhóm em cần chuẩn hóa lại artifact Phase 5 vào workspace để tái lập end-to-end sạch hơn."

#### 4. Hyperparameter search có thể chưa đủ sâu

Code CLI Phase 5 default:

```text
optuna_trials = 3
learning rate = [1e-6, 1e-4]
batch size    = {8, 16, 32}
epochs        = 2..7
max_length    = 128
```

Nếu paper chạy nhiều trial hơn, epoch khác, GPU khác, hoặc checkpoint selection khác, kết quả có thể khác.

Đặc biệt max length 128 có thể truncate abstract/triples khá mạnh. Với paper khoa học, abstract thường dài; phần cuối abstract hoặc triples có thể bị cắt.

Cách trả lời:

> "Hiện tại nhóm em ưu tiên clone pipeline và chạy được toàn bộ ma trận thí nghiệm. Để match paper hơn, cần tăng trial, kiểm soát checkpoint, và kiểm tra ảnh hưởng của max_length."

#### 5. Cách inject cluster signal là implementation choice

Phase 5 hiện không chỉ đưa text vào classifier, mà còn verbalize cluster signal:

```text
clustering mode {rep}. propagated cluster {id}. confidence {score}. [SEP] {base_text}
```

Đây là cách nhóm hiện thực vì paper không mô tả chi tiết cluster propagation sang classifier.

Nếu paper dùng cluster label theo cách khác, ví dụ feature số, special token, hoặc không inject vào input text như repo, thì kết quả sẽ khác.

Cách trả lời:

> "Nhóm em chọn cách đưa cluster signal vào input text để Transformer có thể dùng được mà không đổi architecture. Đây là assumption rõ ràng, không phải chi tiết được paper công bố đầy đủ."

#### 6. Triple quality có thể nhiễu

Triples được extract bằng dependency rules:

- subject: `nsubj`, `nsubjpass`
- relation: `VERB`, `AUX`
- object: `dobj`, `attr`, fallback `prep -> pobj/pcomp`

Rule-based extraction chắc chắn có noise. Ví dụ subject có thể là `we`, `which`, object có thể là phrase không đủ nghĩa. Triple-only rows trong Table 2 thấp hơn rõ:

```text
abstract -> triples : accuracy 0.7445, macro-F1 0.3773
hybrid   -> triples : accuracy 0.7345, macro-F1 0.3589
concat   -> triples : accuracy 0.7295, macro-F1 0.3295
triples  -> triples : accuracy 0.7180, macro-F1 0.3482
```

Điều này khớp với tinh thần paper: triple-only thường kém hơn abstract/hybrid vì mất ngữ cảnh.

### Có phải implementation sai không?

Không thể kết luận đơn giản là sai. Phải tách hai mức:

#### Mức 1: Pipeline implementation

Các thành phần chính đã đúng hướng:

- có Phase 1 data/triples/representations;
- có Phase 2 embeddings;
- có Phase 3 clustering;
- có Phase 4 propagation;
- có Phase 5 classification code;
- có Phase 6 reports.

Về mặt architecture, pipeline bám paper.

#### Mức 2: Experimental reproduction

Kết quả chưa match paper. Đây là reproduction gap. Cần giải thích và xử lý bằng các bước kiểm chứng thêm.

Cách nói với thầy:

> "Implementation pipeline đã đầy đủ, nhưng experimental reproduction chưa match tuyệt đối. Nhóm em đã xác định các nguyên nhân gap và sẽ bổ sung kiểm chứng để thu hẹp sai khác."

### Nếu thầy hỏi cần làm gì để cải thiện classification?

Nên trả lời theo thứ tự ưu tiên:

1. Chuẩn hóa lại Phase 5 artifacts trong `outputs/phase5_classification/`.
2. In label distribution và cân nhắc loại/merge labels quá hiếm.
3. Chạy lại với split stratified có kiểm soát và ghi rõ seed.
4. Tăng Optuna trials, ví dụ 10 hoặc 20.
5. Thử `max_length = 256` hoặc `512` cho classification.
6. Chạy ablation `cluster_signal_mode none/prefix/suffix`.
7. Báo cáo per-label F1 để chứng minh macro-F1 thấp do class hiếm.
8. Tạo top document-level errors để phân tích model nhầm gì.

---

## 2. HDBSCAN: vì sao không claim là tốt nhất?

### Câu hỏi thầy có thể hỏi

"Bảng clustering cho thấy HDBSCAN ARI/NMI rất cao, vậy tại sao lại nói HDBSCAN kém? Tại sao không chọn HDBSCAN là best?"

### Câu trả lời ngắn nên nói

> "HDBSCAN có ARI/NMI cao trong bảng vì metrics hiện được tính trên phần non-noise. Nhưng HDBSCAN gán hơn 90% documents là noise ở nhiều config tốt nhất. Nghĩa là nó chỉ cluster một phần nhỏ rất dễ, còn bỏ qua đa số dữ liệu. Vì vậy không nên claim HDBSCAN tốt nhất trên toàn bộ dataset."

### Kết quả cụ thể trong repo

Top HDBSCAN theo score hiện tại:

```text
abstract + MiniLM + HDBSCAN(min_cluster_size=50)
ARI            : 0.9417
NMI            : 0.8993
Noise fraction : 0.9244
Score          : 0.9080

concatenate + MiniLM + HDBSCAN(min_cluster_size=100)
ARI            : 0.9394
NMI            : 0.8935
Noise fraction : 0.9246

hybrid + MiniLM + HDBSCAN(min_cluster_size=100)
ARI            : 0.9333
NMI            : 0.8865
Noise fraction : 0.9266
```

Noise fraction `0.9244` nghĩa là khoảng `92.44%` documents bị gán label `-1` noise. Trên 5.000 docs, chỉ còn khoảng 378 docs được cluster thật sự.

Vì vậy ARI/NMI cao không có nghĩa là model cluster tốt toàn bộ 5.000 docs. Nó chỉ nói rằng trong phần nhỏ mà HDBSCAN dám cluster, các điểm đó khá pure.

### Vì sao chuyện này xảy ra?

HDBSCAN là density-based clustering:

- Nó tìm vùng có mật độ cao.
- Các điểm không thuộc vùng đủ dày bị gán noise.
- Document embeddings trong không gian cao chiều thường không có density separation rõ như dữ liệu 2D/3D.

Với scientific document embeddings:

- các lĩnh vực giao nhau nhiều;
- abstract có nhiều thuật ngữ chung;
- embeddings có high-dimensional geometry;
- class boundaries không tạo thành vùng density sắc nét.

Do đó HDBSCAN có xu hướng:

- chỉ giữ vài "dense islands" rất chắc;
- bỏ phần lớn documents thành noise.

### Vấn đề trong cách tính metric hiện tại

Trong code `run_hdbscan()`, metric ARI/NMI được tính trên mask:

```python
mask = labels != -1
ari = adjusted_rand_score(y_true[mask], labels[mask])
nmi = normalized_mutual_info_score(y_true[mask], labels[mask])
```

Tức là noise docs bị loại khỏi ARI/NMI.

Cách tính này không sai nếu mục tiêu là đánh giá "quality của phần clustered", nhưng không đủ nếu muốn đánh giá "coverage toàn bộ dataset".

Vì thế cần luôn báo kèm:

- ARI;
- NMI;
- noise fraction;
- số docs non-noise;
- coverage.

### Tại sao paper/plan nói HDBSCAN kém?

Paper/plan quan tâm clustering toàn bộ document set. Nếu một thuật toán gán hơn 90% docs là noise, nó không hữu ích để tổ chức toàn bộ tập tài liệu.

Nói cách khác:

- HDBSCAN có thể tạo vài cụm nhỏ rất pure.
- Nhưng nó không phân cụm được phần lớn dữ liệu.
- Vì vậy về practical clustering, HDBSCAN kém hơn KMeans/GMM.

### Nên report HDBSCAN như thế nào?

Không nên nói:

> "HDBSCAN tốt nhất vì ARI 0.94."

Nên nói:

> "HDBSCAN đạt ARI/NMI cao trên phần non-noise, nhưng noise fraction trên 90%, nên không phù hợp làm kết quả best overall. Với đánh giá thực dụng trên toàn bộ dataset, nhóm ưu tiên KMeans/GMM."

### Nếu cần chọn best clustering để so sánh paper

Nên dùng best non-HDBSCAN hoặc report riêng:

Best non-HDBSCAN hiện tại:

```text
abstract + SciBERT + GMM(k=5)
ARI   : 0.5305
NMI   : 0.5277
Score : 0.5291
```

Một số config tốt khác:

```text
abstract + SPECTER + KMeans(k=4)
ARI   : 0.4949
NMI   : 0.4840

hybrid + SPECTER + KMeans(k=4)
ARI   : 0.4987
NMI   : 0.4795

abstract + SciBERT + KMeans(k=4)
ARI   : 0.4840
NMI   : 0.4920
```

So với expected target:

```text
Paper/plan target ARI : ~0.47
Paper/plan target NMI : ~0.55
Repo non-HDBSCAN ARI  : up to ~0.53
Repo non-HDBSCAN NMI  : up to ~0.53
```

Kết quả này tương đối gần clustering target hơn classification target.

### Nếu thầy hỏi tại sao vẫn chạy HDBSCAN?

Câu trả lời:

> "Vì paper có thử HDBSCAN nên nhóm vẫn reproduce đầy đủ. Nhưng sau khi chạy, nhóm phân tích thêm noise fraction và nhận thấy HDBSCAN không phù hợp để chọn làm best overall. Kết luận này thực ra khớp với nhận xét của paper/plan rằng HDBSCAN kém do nhiều noise."

### Cách sửa để report chặt hơn

Nên bổ sung một trong các cách sau:

1. Tạo bảng `best_non_hdbscan.csv` để so sánh chính với paper.
2. Với HDBSCAN, tính thêm ARI/NMI trên toàn bộ docs, coi `-1` là một cluster.
3. Đặt threshold, ví dụ chỉ xét HDBSCAN nếu `noise_fraction <= 0.5`.
4. Report "coverage-adjusted score":

```text
coverage = 1 - noise_fraction
adjusted_score = (0.5 * ARI + 0.5 * NMI) * coverage
```

5. Vẽ biểu đồ noise fraction theo representation/model.

---

## 3. Phase 4 propagation: assumption của nhóm là gì?

### Câu hỏi thầy có thể hỏi

"Paper có nói rõ cách propagate cluster labels sang classification set không? Nếu không nói rõ, nhóm làm vậy có hợp lý không?"

### Câu trả lời ngắn nên nói

> "Paper có nhắc cluster signal/propagation nhưng không mô tả đủ chi tiết implementation. Vì vậy nhóm em chọn một cách hiện thực minh bạch và dễ tái lập: nearest-neighbor propagation trong embedding space. Đây là assumption của nhóm, đã được ghi rõ trong report."

### Phase 4 đang làm gì?

Phase 4 lấy clusters học từ 5.000 clustering documents và gán cluster signal cho 10.000 classification documents.

Luồng xử lý:

```text
5.000 cluster docs
  -> đã có embeddings
  -> đã có cluster labels từ KMeans/GMM

10.000 classify docs
  -> đã có embeddings trong cùng representation/model
  -> tìm nearest neighbors trong 5.000 cluster docs
  -> lấy cluster labels của neighbors
  -> vote ra propagated_cluster_id
```

Output lưu tại:

```text
outputs/phase4_cluster_propagation/{representation}/{model}/{algorithm_config}/
  propagated_clusters.jsonl
  propagated_clusters.csv
  propagation_summary.json
  propagation_config.json
```

### Chi tiết thuật toán nhóm dùng

Với mỗi classification document:

1. Chọn embedding cùng representation và cùng model với clustering job.
2. Dùng `NearestNeighbors` tìm `k=5` cluster documents gần nhất.
3. Lấy cluster ids của 5 neighbors.
4. Gán cluster bằng `majority_vote`.
5. Nếu hòa, chọn cluster của neighbor gần nhất trong nhóm hòa.
6. Tính confidence = tỷ lệ vote của cluster thắng.
7. Lưu neighbor ids, neighbor distances, neighbor labels để trace.

Default hiện tại:

```text
neighbor_k      : 5
metric          : cosine
assignment_rule : majority_vote
handle_noise    : ignore
seed            : 42
```

### Vì sao cách này hợp lý?

#### 1. Không dùng true label của classification set

Propagation chỉ dùng:

- embeddings;
- cluster labels từ Phase 3;
- nearest-neighbor distances.

Nó không dùng true label của classification docs để gán cluster. Vì vậy không phải label leakage trực tiếp.

Nếu thầy hỏi "có dùng nhãn thật không?", trả lời:

> "Không. Nhãn thật chỉ dùng để evaluate classifier. Propagation chỉ dựa trên embedding similarity và cluster id từ tập clustering."

#### 2. Giữ split non-overlap

Phase 1 đã tách:

- 5.000 docs clustering;
- 10.000 docs classification;

không overlap. Vì vậy document classification không được cluster trực tiếp cùng chính nó trong Phase 3.

Nếu thầy hỏi "sao không cluster cả 15.000 docs?", trả lời:

> "Nhóm em muốn giữ đúng tinh thần tách unsupervised cluster set và supervised classification set. Nếu cluster cả classification set, cluster signal có thể bị phụ thuộc trực tiếp vào evaluation documents hơn."

#### 3. Dễ giải thích và tái lập

Nearest-neighbor propagation đơn giản:

- không thêm model mới;
- không thêm training mới;
- có thể trace từng document được gán cluster vì những neighbors nào;
- có confidence.

Điều này hợp với reproduction vì nhóm không muốn introduce một module quá phức tạp không có trong paper.

#### 4. Cùng embedding space

Mỗi propagation job chạy trong cùng representation/model:

```text
cluster abstract + SciBERT -> classify abstract + SciBERT
cluster hybrid + SPECTER   -> classify hybrid + SPECTER
```

Vì cùng embedding model và representation, nearest-neighbor distance có ý nghĩa hơn.

### Bằng chứng Phase 4 không bị collapse quá mạnh

Workspace hiện có 32 propagation jobs:

```text
4 representations x 4 embedding models x 2 algorithms(KMeans/GMM) = 32 jobs
```

Tổng quan các jobs:

```text
largest_cluster_fraction min/mean/max : 0.219 / 0.3772 / 0.5107
mean_confidence min/mean/max          : 0.783 / 0.8964 / 0.9385
unique propagated clusters min/max    : 3 / 9
```

Điều này cho thấy:

- Không phải mọi job đều collapse vào một cluster duy nhất.
- Một số job vẫn hơi lệch, cluster lớn nhất có thể chiếm khoảng 51%.
- Confidence trung bình cao, nghĩa là neighbors thường đồng thuận.

Ví dụ một job:

```text
abstract + SciBERT + GMM(k=5)
n_classify_docs              : 10000
n_unique_propagated_clusters : 5
largest_cluster_fraction     : 0.4744
mean_confidence              : 0.9298
metric                       : cosine
neighbor_k                   : 5
```

### Phase 4 có phải reproduce chính xác paper không?

Không nên nói là chính xác tuyệt đối, vì paper không mô tả đủ chi tiết.

Nên nói:

> "Đây là phần nhóm em phải đưa ra implementation assumption. Nhóm chọn nearest-neighbor propagation vì nó phù hợp với embedding space, không dùng nhãn thật, dễ trace và dễ reproduce. Nhóm đã ghi rõ assumption này để phân biệt với phần paper mô tả rõ."

### Propagation signal được đưa vào classifier như thế nào?

Phase 5 biến propagated cluster thành text signal:

```text
clustering mode {clustering_representation}.
propagated cluster {cluster_id}.
confidence {propagation_confidence}.
[SEP]
{base_text}
```

Ví dụ:

```text
clustering mode abstract. propagated cluster 3. confidence 0.800. [SEP] this paper proposes ...
```

Vì sao làm vậy?

- `AutoModelForSequenceClassification` nhận input text tokens.
- Nếu muốn thêm numeric cluster id trực tiếp, phải sửa architecture hoặc thêm feature head.
- Verbalization giúp reuse Transformer classifier mà không thay model architecture.

Nhưng đây cũng là một assumption. Nếu paper dùng cluster feature khác, kết quả sẽ khác.

### Rủi ro của cách propagation hiện tại

#### 1. Cluster id là categorical nhưng bị verbalize như text

Transformer có thể hiểu "cluster 3" như một token sequence, nhưng cluster id không có nghĩa ngữ nghĩa tự nhiên.

Rủi ro:

- Model có thể học pattern id-specific.
- Cluster ids giữa các representation/model không có cùng ý nghĩa.

Cách giảm rủi ro:

- Chạy ablation `cluster_signal_mode none`.
- So sánh prefix/suffix/no signal.
- Hoặc thiết kế architecture nhận cluster id như categorical embedding.

#### 2. Propagation phụ thuộc chất lượng clustering

Nếu Phase 3 cluster không tốt, propagated signal có thể nhiễu và làm classifier tệ hơn.

Cách kiểm tra:

- So sánh classification với/không cluster signal.
- Chỉ dùng best KMeans/GMM có ARI/NMI ổn định.
- Không dùng HDBSCAN noise-heavy cho propagation chính.

#### 3. Nearest neighbors có thể bias theo class lớn

Vì class `cs` và `math` nhiều, nearest neighbors dễ rơi vào các vùng đông của class lớn.

Cách kiểm tra:

- Xem propagated cluster distribution.
- Xem per-label F1.
- Xem top errors ở labels hiếm.

### Nếu thầy hỏi "vậy Phase 4 có cần thiết không?"

Câu trả lời:

> "Có, vì nó là cách nối Phase 3 unsupervised với Phase 5 supervised. Nếu không có Phase 4, clustering chỉ là một thí nghiệm riêng, không đóng góp tín hiệu cho classifier. Phase 4 giúp kiểm tra giả thuyết: cấu trúc cụm trong embedding space có giúp classification không."

Nhưng cũng nên nói thêm:

> "Tuy nhiên, vì paper không mô tả chi tiết, nhóm em xem Phase 4 là assumption cần ablation. Cần so sánh với baseline không dùng cluster signal để biết propagation có thật sự giúp không."

### Nếu thầy hỏi "tại sao không dùng HDBSCAN propagation?"

Câu trả lời:

> "Do HDBSCAN tạo quá nhiều noise, nhiều config trên 90% documents là noise. Nếu propagate HDBSCAN, rất nhiều classification docs có thể nhận signal noise hoặc signal không ổn định. Vì vậy nhóm ưu tiên KMeans/GMM cho propagation chính."

### Các cải tiến có thể làm tiếp cho Phase 4

1. Distance-weighted vote thay vì majority vote.
2. Chỉ propagate nếu confidence vượt threshold, ví dụ `>= 0.6`.
3. Dùng top-k khác: `k=1`, `k=3`, `k=5`, `k=10`.
4. Chuẩn hóa cluster signal thành special token:

```text
[CLUSTER_ABSTRACT_3] [SEP] base text
```

5. Thêm categorical embedding cho cluster id thay vì verbalize text.
6. Ablation no-propagation baseline.
7. Report per-label effect: label nào được cluster signal giúp, label nào bị hại.

---

## 4. Câu trả lời mẫu khi báo cáo

### Nếu thầy hỏi tổng quan gap

> "Nhóm em đã hoàn thành implementation pipeline theo các phase của paper. Tuy nhiên, nhóm em tách rõ giữa 'pipeline đã clone xong' và 'kết quả đã match paper'. Hiện pipeline chạy được và có artifacts, nhưng classification còn gap so với paper. Nhóm em đã xác định các nguyên nhân chính: dataset/filtering chưa thể giống tuyệt đối vì paper không công bố rõ, label distribution hiện rất imbalance, Phase 5 artifacts cần chuẩn hóa lại, và cluster propagation là một assumption implementation."

### Nếu thầy hỏi classification có thất bại không?

> "Không hẳn là thất bại. Accuracy và weighted-F1 khoảng 0.81 cho thấy model vẫn học được class lớn. Nhưng macro-F1 thấp cho thấy vấn đề ở class hiếm. Đây là kết quả có giá trị phân tích, vì nó chỉ ra reproduction đang khác paper ở label distribution hoặc training setup. Nhóm em chưa claim match paper, mà xem đây là reproduction gap cần xử lý."

### Nếu thầy hỏi vì sao macro-F1 thấp

> "Vì split hiện tại có 21 labels và imbalance rất mạnh. Hai lớp `cs` và `math` chiếm khoảng 54.9% dữ liệu, trong khi 9 labels có dưới 100 documents. Macro-F1 tính đều cho mỗi class nên các class hiếm kéo điểm xuống mạnh. Accuracy/weighted-F1 cao hơn vì bị chi phối bởi class lớn."

### Nếu thầy hỏi HDBSCAN

> "HDBSCAN nhìn rất cao nếu chỉ xem ARI/NMI, nhưng noise fraction hơn 90%. Code hiện tính ARI/NMI trên phần non-noise, nên điểm cao chỉ phản ánh một phần nhỏ documents được cluster rất pure. Vì vậy nhóm em không chọn HDBSCAN làm best overall, mà report nó như một thuật toán có coverage thấp, khớp với nhận xét paper rằng density-based clustering không phù hợp lắm cho embedding space này."

### Nếu thầy hỏi Phase 4 có tự nghĩ ra không?

> "Dạ, phần chi tiết propagation là assumption của nhóm vì paper không mô tả đủ cụ thể. Nhóm chọn nearest-neighbor propagation vì nó minh bạch, không dùng label thật của classification set, giữ split non-overlap, và có thể trace bằng neighbor ids/distances/confidence. Nhóm sẽ ghi rõ đây là implementation assumption chứ không phải chi tiết paper công bố đầy đủ."

### Nếu thầy hỏi cần làm gì tiếp

> "Các bước tiếp theo là chuẩn hóa lại Phase 5 raw artifacts vào repo, sửa Phase 6 để đọc đúng path, tạo confusion matrix/top document errors thật, thêm bảng best non-HDBSCAN, và chạy ablation không dùng cluster signal để định lượng Phase 4 có giúp classifier hay không."

---

## 5. Checklist slide nên có

Nếu làm slide báo cáo, nên có 3 slide riêng:

### Slide 1: Classification gap

Nên ghi:

```text
Paper target : Acc ~92.6%, Macro-F1 ~0.925
Repo current : Acc 81.4%, Macro-F1 0.531

Likely causes:
- Different dataset filtering / recent-years assumption
- 21 labels, highly imbalanced
- 9 labels < 100 docs
- Phase 5 artifacts need reproducibility cleanup
- Cluster signal injection is implementation-specific
```

### Slide 2: HDBSCAN caveat

Nên ghi:

```text
HDBSCAN top ARI/NMI:
ARI 0.9417, NMI 0.8993
but noise_fraction 0.9244

Interpretation:
- High purity on small non-noise subset
- Not good full-dataset clustering
- Prefer KMeans/GMM for main comparison
```

### Slide 3: Propagation assumption

Nên ghi:

```text
Paper: cluster signal mentioned, exact implementation underspecified
Our assumption: nearest-neighbor propagation

For each classify doc:
1. find top-5 nearest cluster docs
2. majority vote cluster IDs
3. store confidence/distances/neighbors
4. inject cluster signal into classifier text

No true labels used in propagation.
```

---

## 6. Kết luận nên chốt với thầy

Kết luận nên nói:

> "Nhóm em đã hoàn thành implementation pipeline đầy đủ theo paper, nhưng đang ở giai đoạn phân tích reproduction gap. Phần clustering tương đối gần target nếu xét KMeans/GMM, HDBSCAN cần report kèm noise. Phần classification chưa match paper, chủ yếu do dataset/filtering/imbalance và một số assumption implementation. Nhóm em đã xác định rõ các gap này và có kế hoạch chỉnh để báo cáo cuối minh bạch hơn."

Không nên nói:

> "Nhóm em đã reproduce hoàn toàn giống paper."

Nên nói:

> "Nhóm em đã reproduce pipeline và đang chuẩn hóa kết quả/giải thích sai khác so với paper."

