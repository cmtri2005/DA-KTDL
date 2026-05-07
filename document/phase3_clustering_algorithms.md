# Phase 3 – Chi tiết Thuật toán Clustering

> Tài liệu tham chiếu cho pipeline tái hiện paper  
> **"Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents"**  
> Phạm vi: 5.000 tài liệu clustering, 4 representation × 4 embedding model = 16 combination.

---

## Tổng quan so sánh

| Thuật toán | Loại | Số cụm | Dạng cụm | Xử lý outlier | Soft assignment |
|---|---|---|---|---|---|
| **KMeans** | Partition-based | Cố định `k` | Spherical (hình cầu) | Không | Không |
| **GMM** | Model-based (probabilistic) | Cố định `k` | Elliptical (elip tùy ý) | Không | Có (xác suất) |
| **HDBSCAN** | Density-based | Tự động | Hình dạng tùy ý | Có (noise label = -1) | Có (soft clustering) |

---

## 1. K-Means Clustering

### 1.1 Ý tưởng cốt lõi

KMeans chia `n` điểm dữ liệu thành `k` cụm sao cho tổng bình phương khoảng cách từ mỗi điểm đến centroid của cụm tương ứng (gọi là **Within-Cluster Sum of Squares – WCSS** hay **Inertia**) là nhỏ nhất.

$$\text{minimize} \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$

Trong đó:
- $C_i$: tập hợp các điểm thuộc cụm $i$
- $\mu_i$: centroid (trung bình) của cụm $i$
- $\|\cdot\|^2$: bình phương khoảng cách Euclidean

### 1.2 Thuật toán Lloyd's (Standard KMeans)

```
1. Khởi tạo k centroid (random hoặc KMeans++)
2. Lặp cho đến khi hội tụ:
   a. Assignment step: Gán mỗi điểm vào cụm có centroid gần nhất
   b. Update step: Tính lại centroid = trung bình của tất cả điểm trong cụm
3. Hội tụ khi centroid không thay đổi (hoặc thay đổi < tol)
```

### 1.3 Khởi tạo KMeans++ (sklearn default)

KMeans++ chọn centroid ban đầu thông minh hơn, tránh hội tụ tại local minima tệ:

```
1. Chọn centroid đầu tiên ngẫu nhiên
2. Với mỗi centroid tiếp theo:
   - Tính D(x)² = khoảng cách bình phương từ x đến centroid gần nhất đã chọn
   - Chọn điểm tiếp theo theo xác suất tỉ lệ D(x)²
3. Lặp đến khi có đủ k centroid
```

Lợi ích: giảm WCSS cuối cùng, hội tụ nhanh hơn và ổn định hơn.

### 1.4 Tham số quan trọng (sklearn `KMeans`)

| Tham số | Giá trị mặc định | Ý nghĩa | Giá trị dùng trong Phase 3 |
|---|---|---|---|
| `n_clusters` | `8` | Số cụm `k` | **Sweep từ 3 đến 12** |
| `init` | `'k-means++'` | Phương pháp khởi tạo centroid | `'k-means++'` |
| `n_init` | `10` | Số lần chạy lại với centroid ngẫu nhiên khác nhau | `10` (hoặc `20` để ổn định hơn) |
| `max_iter` | `300` | Số vòng lặp tối đa mỗi lần chạy | `300` |
| `tol` | `1e-4` | Ngưỡng hội tụ (thay đổi centroid) | `1e-4` |
| `random_state` | `None` | Seed cho reproducibility | **`42`** |
| `algorithm` | `'lloyd'` | Thuật toán (`'lloyd'` hoặc `'elkan'`) | `'lloyd'` |

> **Lưu ý về `n_init`**: sklearn ≥ 1.2 đổi default `n_init` từ `10` → `'auto'` (= 10 với `k-means++`, = 1 với `random`). Nên set tường minh.

### 1.5 Giả định và hạn chế

| Giả định | Thực tế trong project |
|---|---|
| Cụm có hình cầu đồng kích thước | Embedding text ≠ spherical → có thể bị bias |
| Số cụm `k` phải biết trước | Sweep `k ∈ [3, 12]` để tìm best `k` |
| Nhạy với outlier | Các embedding cực trị kéo centroid |
| Giả định phương sai đều giữa các cụm | Bị vi phạm với text đa dạng ngữ nghĩa |

### 1.6 Tại sao dùng KMeans trong paper

- **Nhanh và scalable**: $O(n \cdot k \cdot d \cdot i)$ với $n$ = số doc, $d$ = chiều embedding, $i$ = số iteration.
- **Baseline mạnh**: thường cho ARI/NMI tốt khi embedding tốt (ví dụ MPNet + Abstract).
- **Dễ tái hiện**: chỉ cần cố định `random_state=42`.

---

## 2. Gaussian Mixture Model (GMM)

### 2.1 Ý tưởng cốt lõi

GMM giả định dữ liệu được sinh ra từ hỗn hợp `k` phân phối Gaussian. Mỗi Gaussian $\mathcal{N}(\mu_i, \Sigma_i)$ đại diện cho một cụm.

$$p(x) = \sum_{i=1}^{k} \pi_i \cdot \mathcal{N}(x \mid \mu_i, \Sigma_i)$$

Trong đó:
- $\pi_i$: mixing weight (tỷ lệ điểm thuộc Gaussian $i$), $\sum \pi_i = 1$
- $\mu_i$: vector trung bình của Gaussian thứ $i$
- $\Sigma_i$: ma trận hiệp phương sai (covariance matrix)

### 2.2 Thuật toán EM (Expectation-Maximization)

GMM được tối ưu bằng EM, gồm 2 bước lặp:

**E-step (Expectation)** – Tính trách nhiệm (responsibility):
$$r_{ij} = \frac{\pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}{\sum_{l=1}^{k} \pi_l \cdot \mathcal{N}(x_i \mid \mu_l, \Sigma_l)}$$

$r_{ij}$ = xác suất điểm $x_i$ thuộc Gaussian $j$ (soft assignment).

**M-step (Maximization)** – Cập nhật tham số:
$$\pi_j = \frac{N_j}{N}, \quad \mu_j = \frac{1}{N_j} \sum_i r_{ij} x_i, \quad \Sigma_j = \frac{1}{N_j} \sum_i r_{ij}(x_i - \mu_j)(x_i - \mu_j)^T$$

Lặp E→M đến khi log-likelihood hội tụ.

### 2.3 Dạng Covariance Matrix (`covariance_type`)

Đây là tham số quan trọng nhất ảnh hưởng đến hình dạng cụm:

| `covariance_type` | Ý nghĩa | Số tham số | Hình dạng cụm |
|---|---|---|---|
| `'full'` | Mỗi Gaussian có $\Sigma$ riêng, không giới hạn | Nhiều nhất | Elip tùy ý, hướng tùy ý |
| `'tied'` | Tất cả Gaussian dùng chung một $\Sigma$ | Trung bình | Elip cùng hướng và kích thước |
| `'diag'` | $\Sigma$ là ma trận đường chéo (không tương quan) | Ít hơn | Elip căn thẳng theo trục tọa độ |
| `'spherical'` | $\Sigma = \sigma^2 I$ (scalar) | Ít nhất | Hình cầu (tương đương KMeans về hình dạng) |

> **Trong paper và Phase 3**: dùng `covariance_type='full'` để tận dụng hết sức mạnh của GMM, nhưng chú ý với embedding chiều cao (384–768 chiều), `full` dễ bị **singular matrix** → cần regularization.

### 2.4 Tham số quan trọng (sklearn `GaussianMixture`)

| Tham số | Giá trị mặc định | Ý nghĩa | Giá trị dùng trong Phase 3 |
|---|---|---|---|
| `n_components` | `1` | Số Gaussian `k` | **Sweep từ 3 đến 12** |
| `covariance_type` | `'full'` | Dạng covariance | `'full'` |
| `tol` | `1e-3` | Ngưỡng hội tụ EM | `1e-3` |
| `reg_covar` | `1e-6` | Regularization thêm vào đường chéo $\Sigma$ (tránh singular) | `1e-6` (tăng nếu `ConvergenceWarning`) |
| `max_iter` | `100` | Số vòng lặp EM tối đa | `200` |
| `n_init` | `1` | Số lần khởi tạo ngẫu nhiên | `5` (để ổn định) |
| `init_params` | `'kmeans'` | Khởi tạo tham số (`'kmeans'` / `'k-means++'` / `'random'`) | `'k-means++'` |
| `random_state` | `None` | Seed | **`42`** |

### 2.5 Sự khác biệt GMM vs KMeans

| Đặc điểm | KMeans | GMM |
|---|---|---|
| Assignment | Hard (mỗi điểm thuộc đúng 1 cụm) | Soft (xác suất thuộc từng cụm) |
| Hình cụm | Chỉ spherical | Bất kỳ (tùy `covariance_type`) |
| Tối ưu hóa | Minimize WCSS | Maximize log-likelihood |
| Uncertainty | Không | Có (posterior probability) |
| Chi phí tính toán | Thấp hơn | Cao hơn (do EM) |

### 2.6 Tại sao dùng GMM trong paper

- Mô hình hóa linh hoạt hơn KMeans nhờ covariance tùy ý.
- Cho phép cụm có kích thước và hình dạng khác nhau (phù hợp với topic đa dạng như astro-ph vs cs vs math).
- Với embedding tốt, GMM thường cho ARI/NMI tương đương hoặc tốt hơn KMeans nhẹ.

---

## 3. HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

### 3.1 Ý tưởng cốt lõi

HDBSCAN là phiên bản phân cấp (hierarchical) của DBSCAN. Thay vì yêu cầu tham số ngưỡng mật độ cố định $\varepsilon$, HDBSCAN tự động tìm cấu trúc phân cụm ở **nhiều mức mật độ** khác nhau, sau đó chọn ra các cụm **ổn định nhất**.

**Ba bước chính:**

```
1. Core distance  →  mutual reachability distance
2. Minimum spanning tree (MST) trên mutual reachability graph
3. Extract hierarchy (dendrogram) → chọn cụm ổn định
```

### 3.2 Các khái niệm cốt lõi

#### Core Distance
$$d_{\text{core}}(p, k) = d(p, N_k(p))$$
Khoảng cách từ điểm $p$ đến điểm láng giềng thứ $k$ gần nhất (tham số `min_samples`).

#### Mutual Reachability Distance
$$d_{\text{mreach}}(p, q) = \max\left(d_{\text{core}}(p), d_{\text{core}}(q), d(p, q)\right)$$
"Làm mượt" khoảng cách để đảm bảo điểm thưa được đo từ góc nhìn mật độ cục bộ.

#### HDBSCAN Hierarchy
- Xây dựng MST trên đồ thị mutual reachability.
- Rút trích dendrogram từ MST.
- Tính **cluster stability** (độ ổn định) cho mỗi nhánh.
- Chọn cụm bằng cách tối đa hóa tổng stability theo thuật toán Ecut.

### 3.3 Tham số quan trọng (thư viện `hdbscan`)

| Tham số | Mặc định | Ý nghĩa | Giá trị sweep trong Phase 3 |
|---|---|---|---|
| `min_cluster_size` | `5` | **Quan trọng nhất.** Số điểm tối thiểu để một nhóm được coi là cụm | **Sweep: 5, 10, 20, 30, 50, 100** |
| `min_samples` | `None` → bằng `min_cluster_size` | Ảnh hưởng đến core distance; nhỏ hơn → ít noise hơn | `None` (default) hoặc sweep riêng |
| `cluster_selection_epsilon` | `0.0` | Ngưỡng khoảng cách tối thiểu để merge cụm (DBSCAN-like epsilon) | `0.0` (không dùng) |
| `metric` | `'euclidean'` | Hàm khoảng cách | `'euclidean'` (vì embedding đã L2-normalize) |
| `cluster_selection_method` | `'eom'` | Phương pháp chọn cụm: `'eom'` (Excess of Mass) hoặc `'leaf'` | `'eom'` |
| `alpha` | `1.0` | Tham số nhỏ ảnh hưởng đến split condition | `1.0` |
| `prediction_data` | `False` | Cần bật nếu dùng soft clustering | `True` nếu muốn soft label |

> **Lưu ý**: thư viện `hdbscan` (standalone) và `sklearn.cluster.HDBSCAN` (sklearn ≥ 1.3) có API hơi khác nhau. Project dùng `hdbscan` standalone (đã cài trong requirements).

### 3.4 Xử lý điểm Noise (label = -1)

HDBSCAN gán `label = -1` cho các điểm không thuộc cụm nào (outlier). Đây là đặc trưng quan trọng:

- **Noise fraction** = `sum(labels == -1) / n_docs`
- Paper ghi nhận HDBSCAN kém trong không gian cao chiều vì text embedding **không có sự phân tách mật độ rõ ràng** → noise fraction cao.
- Trong Phase 3, noise fraction là metric phụ trừ điểm:  
  `score_HDBSCAN = NMI + 0.5 * ARI - 0.5 * noise_fraction`

### 3.5 Soft Clustering với HDBSCAN

Nếu bật `prediction_data=True`, có thể lấy xác suất thành viên:

```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, prediction_data=True)
clusterer.fit(X)
soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
# soft_clusters[i] = vector xác suất thuộc từng cụm của điểm i
```

### 3.6 Tại sao HDBSCAN kém hơn trong paper

| Nguyên nhân | Giải thích |
|---|---|
| Curse of dimensionality | Embedding 384–768 chiều → mọi điểm đều "gần bằng nhau" theo Euclidean |
| Không gian text không phân cụm theo mật độ | Các chủ đề khoa học overlap nhiều → không có vùng thưa rõ ràng |
| Số lượng cụm không xác định | Dẫn đến noise fraction cao hoặc cụm quá ít/nhiều |
| Nhạy với `min_cluster_size` | Cần sweep kỹ, kết quả dao động lớn |

---

## 4. Metrics Đánh giá Clustering

### 4.1 ARI – Adjusted Rand Index

$$\text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}$$

- **Ý nghĩa**: Đo mức độ đồng thuận giữa phân cụm thuật toán và nhãn thật, đã điều chỉnh cho trường hợp ngẫu nhiên.
- **Phạm vi**: $[-1, 1]$. ARI = 1 → hoàn hảo; ARI = 0 → ngẫu nhiên; ARI < 0 → tệ hơn ngẫu nhiên.
- **Đặc điểm**: Không phụ thuộc vào số cụm, đối xứng (bỏ qua permutation).
- **Hàm sklearn**: `sklearn.metrics.adjusted_rand_score(y_true, y_pred)`

### 4.2 NMI – Normalized Mutual Information

$$\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}$$

- **Ý nghĩa**: Đo lượng thông tin chung giữa phân cụm dự đoán và nhãn thật, chuẩn hóa về [0, 1].
- **Phạm vi**: $[0, 1]$. NMI = 1 → hoàn hảo; NMI = 0 → độc lập.
- **Đặc điểm**: Không bị penalty khi số cụm khác nhau (không như RI thô).
- **Hàm sklearn**: `sklearn.metrics.normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')`

### 4.3 Silhouette Score

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- $a(i)$: khoảng cách trung bình từ $i$ đến các điểm **cùng cụm** (cohesion).
- $b(i)$: khoảng cách trung bình nhỏ nhất từ $i$ đến điểm **cụm gần nhất khác** (separation).
- **Phạm vi**: $[-1, 1]$. Cao → cụm gọn, tách biệt tốt.
- **Lưu ý**: Tính trung bình trên tất cả điểm; với HDBSCAN, bỏ qua noise points (label = -1).
- **Hàm sklearn**: `sklearn.metrics.silhouette_score(X, labels, metric='euclidean')`

> **Silhouette với embedding L2-normalize**: Có thể dùng `metric='cosine'` thay `'euclidean'` vì embedding đã chuẩn hóa. Cosine distance = 1 - cosine similarity.

### 4.4 Noise Fraction (chỉ HDBSCAN)

$$\text{noise\_fraction} = \frac{\sum_{i} \mathbb{1}[\text{label}_i = -1]}{N}$$

- **Phạm vi**: $[0, 1]$. Càng thấp càng tốt.
- **Ý nghĩa**: Tỷ lệ điểm không được gán vào cụm nào.

### 4.5 Tiêu chí chọn Best Configuration

**Best KMeans / GMM** (theo paper):
$$\text{score} = 0.5 \times \text{ARI} + 0.5 \times \text{NMI}$$

**Best HDBSCAN** (theo paper):
$$\text{score} = \text{NMI} + 0.5 \times \text{ARI} - 0.5 \times \text{noise\_fraction}$$

---

## 5. Lưu ý Implementation cho Phase 3

### 5.1 Số nhãn thật (ground truth)

Từ Phase 1, `label` là category cấp đầu tiên của arXiv, ví dụ:
- `cs` (Computer Science)
- `math` (Mathematics)
- `astro-ph` (Astrophysics)
- `cond-mat`, `quant-ph`, `hep-th`, `hep-ph`, `gr-qc`, v.v.

Số nhãn thực tế phụ thuộc vào snapshot. Cần kiểm tra để xác định `k_true` làm tham chiếu cho sweep range `[3, 12]`.

### 5.2 Handling HDBSCAN noise khi tính metrics

```python
mask = labels != -1  # Bỏ qua noise points
ari = adjusted_rand_score(y_true[mask], labels[mask])
nmi = normalized_mutual_info_score(y_true[mask], labels[mask])
# Silhouette tính trên tất cả hoặc chỉ non-noise
sil = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else 0.0
noise_frac = (~mask).mean()
```

### 5.3 Chiều cao của embedding và PCA/UMAP

- Embedding từ MiniLM: **384 chiều**
- Embedding từ MPNet, SciBERT, SPECTER: **768 chiều**
- **KMeans/GMM**: Có thể chạy trực tiếp trên embedding gốc.
- **HDBSCAN**: Rất nhạy với chiều cao (curse of dimensionality). Paper không đề cập giảm chiều. Nên thử cả **có** và **không có** UMAP.
- Nếu dùng UMAP: `umap.UMAP(n_components=50, random_state=42)` trước HDBSCAN.

### 5.4 GMM và singular covariance

Với embedding 768 chiều và `n_docs = 5000`, ma trận covariance `full` có thể bị singular. Giải pháp:
- Tăng `reg_covar` từ `1e-6` lên `1e-3`.
- Dùng `covariance_type='diag'` thay thế nếu vẫn thất bại.
- Giảm chiều trước bằng PCA: `sklearn.decomposition.PCA(n_components=100)`.

### 5.5 Reproducibility

```python
import numpy as np
SEED = 42
np.random.seed(SEED)

# KMeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=SEED)

# GMM
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=k, covariance_type='full', 
                       n_init=5, init_params='k-means++', random_state=SEED)

# HDBSCAN
import hdbscan
hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, metric='euclidean', 
                        cluster_selection_method='eom')
```

### 5.6 Cấu trúc output đề xuất

```
outputs/phase3_clustering/
├── results_table.csv          # ARI, NMI, Silhouette, noise_fraction cho mọi combination
├── results_table_best.csv     # Best config cho mỗi (split, repr, model)
├── {split}/{repr}/{model}/
│   ├── kmeans_k{k}_labels.npy
│   ├── gmm_k{k}_labels.npy
│   ├── hdbscan_mcs{mcs}_labels.npy
│   └── run_config.json
└── cluster_analysis/
    ├── label_distribution.csv # Phân phối nhãn thật trong từng cluster
    └── cluster_purity.csv     # Purity per cluster
```

---

## 6. Kết quả Mong đợi (Expected Targets từ Paper)

| Configuration | ARI | NMI | Ghi chú |
|---|---|---|---|
| Abstract + MPNet + KMeans | ~0.47 | ~0.55 | Best clustering |
| Abstract + MPNet + GMM | ~0.45 | ~0.53 | Tương đương KMeans |
| Triples-only + bất kỳ | Thấp hơn rõ | Thấp hơn rõ | Thiếu ngữ cảnh |
| HDBSCAN bất kỳ | Thấp | Thấp | Nhiều noise |

### Phân tích cụm mong đợi

- **astro-ph cluster**: Thuần nhất cao, tách biệt tốt.
- **math cluster**: Khá thuần nhất.
- **cond-mat vs quant-ph**: Overlap do liên quan vật lý.
- **hep-ph / hep-th / gr-qc**: Overlap cao, cùng lĩnh vực vật lý hạt nhân.
- **Mixed interdisciplinary**: cs và math thường lẫn nhau.

---

## 7. Tóm tắt Workflow Phase 3

```
Với mỗi (representation, embedding_model) trong 4×4 = 16 combo:
  Load embeddings.npy từ Phase 2 output
  
  # KMeans sweep
  Với mỗi k trong [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    Fit KMeans(k, seed=42)
    Tính ARI, NMI, Silhouette
    Lưu kết quả
  
  # GMM sweep
  Với mỗi k trong [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    Fit GMM(k, seed=42)
    Tính ARI, NMI, Silhouette
    Lưu kết quả
  
  # HDBSCAN sweep
  Với mỗi mcs trong [5, 10, 20, 30, 50, 100]:
    Fit HDBSCAN(mcs)
    Tính ARI, NMI, Silhouette, noise_fraction
    Lưu kết quả
  
  # Chọn best
  Best KMeans/GMM = argmax(0.5*ARI + 0.5*NMI)
  Best HDBSCAN = argmax(NMI + 0.5*ARI - 0.5*noise_fraction)

Lưu bảng kết quả → Table 1 equivalent
Lưu cluster labels của best configuration
```

---

*Tài liệu được tạo ngày 06/05/2026 phục vụ Phase 3 của pipeline tái hiện paper.*
