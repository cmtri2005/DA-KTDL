# LƯỢNG HÓA VÀ QUY TRÌNH XỬ LÝ SAU LATE FUSION (POST-LATE FUSION WORKFLOW)

Tài liệu này trình bày chi tiết quy trình xử lý dữ liệu, tối ưu hóa mô hình và hướng ứng dụng thực tế của đồ án kể từ dấu mốc **Late Fusion** trở đi.

---

## 1. Sơ đồ Workflow Tổng thể (Mermaid Diagram)

Dưới đây là sơ đồ chi tiết biểu diễn luồng hoạt động sau bước **Late Fusion**. Quy trình được chia làm 2 nhánh chính chạy song song: **Nhánh Phân loại & Huấn luyện (Classification & Training)** và **Nhánh Ứng dụng thực tế (Downstream Applications)**.

```mermaid
graph TD
    classDef main fill:#ffeaa7,stroke:#fdcb6e,stroke-width:2px;
    classDef mlp fill:#fab1a0,stroke:#e17055,stroke-width:2px;
    classDef loss fill:#ff7675,stroke:#d63031,stroke-width:2px;
    classDef opt fill:#a29bfe,stroke:#6c5ce7,stroke-width:2px;
    classDef app fill:#81ecec,stroke:#00cec9,stroke-width:2px;
    classDef db fill:#74b9ff,stroke:#0984e3,stroke-width:2px;

    LF["<b>Late Fusion Node</b><br/>v_joint = [v_text || v_graph] in R^1024"]:::main
    
    %% Nhánh 1: Phân loại và Huấn luyện
    LF -->|1. Forward Pass| MLP["<b>MLP Classifier Head</b><br/>Linear(1024->256) -> ReLU -> Dropout(0.3) -> Linear(256->21)"]:::mlp
    MLP --> Logits["<b>Raw Logits</b><br/>Z in R^21"]:::mlp
    
    Logits -->|Training Mode| Loss["<b>Cross-Entropy Loss</b><br/>So sánh với Ground Truth Labels"]:::loss
    Loss -->|Backpropagation| Opt["<b>Differential Optimizer (AdamW)</b><br/>Mô hình cập nhật lan truyền ngược"]:::opt
    Opt -->|SciBERT (LR: 2e-5)| SciBERT["SciBERT Weights (Fine-Tuning)"]
    Opt -->|GNN & MLP (LR: 5e-4)| GNN_MLP["GNN & MLP Classifier Weights"]

    Logits -->|Inference Mode| Softmax["<b>Softmax & Argmax</b><br/>Dự đoán nhãn (1 trong 21 lớp)"]:::mlp
    Softmax --> Predict["Predicted arXiv Category"]:::mlp

    %% Nhánh 2: Hướng Ứng dụng Thực tế (Downstream)
    LF -->|2. Embedding Extraction| Extract["<b>Joint Semantic Representation</b><br/>Trích xuất vector đặc trưng 1024-dim"]:::app
    Extract --> Index["<b>Vector Database Indexing</b><br/>Lưu trữ chỉ mục (FAISS / Milvus / Pinecone)"]:::db
    
    Index --> App1["<b>Semantic Document Search</b><br/>Tìm kiếm tài liệu bằng Cosine Similarity"]:::app
    Index --> App2["<b>Academic Recommendation</b><br/>Gợi ý bài báo khoa học tương đồng"]:::app
    
    Extract --> Reduction["<b>Dimensionality Reduction</b><br/>Giảm chiều dữ liệu (t-SNE / UMAP -> 2D)"]:::app
    Reduction --> Visualize["<b>Academic Knowledge Map</b><br/>Trực quan hóa cụm tri thức & ngành học"]:::app
```

---

## 2. Giải thích Chi tiết các Thành phần trong Workflow

### 2.1. Late Fusion Node (Điểm Khởi đầu)
Ghép nối trực tiếp vector biểu diễn ngữ cảnh tuyến tính $v_{text} \in \mathbb{R}^{768}$ từ token `[CLS]` của SciBERT và vector biểu diễn cấu trúc topo đồ thị tri thức $v_{graph} \in \mathbb{R}^{256}$ sau khi gom tụ (Mean + Max Pooling) từ GNN:
$$v_{joint} = [v_{text} \parallel v_{graph}] \in \mathbb{R}^{1024}$$

---

## NHÁNH 1: LUỒNG PHÂN LOẠI & HUẤN LUYỆN (CLASSIFICATION & TRAINING)

### 2.2. Bộ phân loại MLP (MLP Classifier Head)
Vector kết hợp $v_{joint}$ có kích thước lớn ($1024$ chiều) được đưa qua mạng MLP 2 tầng nhằm ánh xạ phi tuyến vào không gian 21 lớp nhãn arXiv:
1. **Fully Connected Layer 1 (FC1):** Biến đổi tuyến tính $W_1 \cdot v_{joint} + b_1$ để đưa không gian đặc trưng từ $1024$ chiều xuống còn $256$ chiều.
2. **Hàm kích hoạt (Activation):** Sử dụng hàm **ReLU (Rectified Linear Unit)** để tạo tính phi tuyến cho mô hình, giúp học các mối quan hệ phức tạp giữa văn bản và đồ thị.
3. **Lớp chống quá khớp (Dropout):** Với xác suất loại bỏ $p = 0.3$, giúp triệt tiêu hiện tượng đồng thích ứng (co-adaptation) của các nơ-ron, tăng tính tổng quát hóa của mô hình trên tập kiểm thử.
4. **Fully Connected Layer 2 (FC2):** Biến đổi tuyến tính $W_2 \cdot h_{mlp} + b_2$ đưa từ $256$ chiều xuống $21$ chiều, cho ra vector **Raw Logits** $Z \in \mathbb{R}^{21}$.

### 2.3. Trạng thái huấn luyện (Training Mode)
* **Cross-Entropy Loss:** Tính toán sai số giữa phân phối dự đoán của Logits $Z$ và phân phối thực tế (nhãn một nóng - One-hot):
  $$\mathcal{L} = -\sum_{c=1}^{21} y_c \log(\hat{y}_c)$$
* **Tối ưu hóa vi phân (Differential Optimizer - AdamW):** Do kiến trúc đa phương thức chứa cả Transformer lớn đã tiền huấn luyện và mạng GNN nhỏ tự xây dựng, chúng ta áp dụng tốc độ học (Learning Rate) khác nhau:
  * **Nhánh SciBERT:** Cực kỳ nhỏ ($2 \times 10^{-5}$) để bảo toàn tri thức ngôn ngữ, chỉ tinh chỉnh nhẹ (fine-tuning) theo miền dữ liệu arXiv.
  * **Nhánh GNN & MLP Classifier:** Lớn hơn ($5 \times 10^{-4}$) để cập nhật nhanh chóng các tham số mới khởi tạo theo không gian vector liên kết.
* **Cập nhật trọng số:** Lan truyền ngược sai số qua toàn bộ mạng để điều chỉnh trọng số của SciBERT, GNN, và MLP Head đồng thời.

### 2.4. Trạng thái suy luận (Inference Mode)
* Áp dụng hàm **Softmax** chuyển đổi Logits thành phân phối xác suất thực sự và dùng hàm **Argmax** để đưa ra nhãn dự đoán cuối cùng:
  $$\hat{y}_{class} = \text{argmax} \left( \text{Softmax}(Z) \right)$$

---

## NHÁNH 2: LUỒNG ỨNG DỤNG THỰC TẾ (DOWNSTREAM APPLICATIONS)

Điểm đắt giá của Đồ án nằm ở chỗ: **Sau khi Late Fusion thành công, chúng ta sở hữu một không gian vector biểu diễn đặc trưng đa phương thức (Joint Embedding Space) vô cùng mạnh mẽ.** Vector $v_{joint}$ này được ứng dụng vào 3 bài toán thực tiễn:

### 2.5. Cơ sở dữ liệu Vector & Tìm kiếm Ngữ nghĩa (Vector DB & Semantic Search)
* **Quy trình:** Đưa toàn bộ các bài viết khoa học qua mô hình Late Fusion đã huấn luyện để trích xuất ra các vector đặc trưng $1024$ chiều. Lưu trữ các vector này vào một **Vector Database** (như FAISS, Milvus, hoặc Pinecone) kèm theo chỉ mục tìm kiếm nhanh (HNSW, IVF-FLAT).
* **Ứng dụng:** Khi người dùng nhập câu hỏi tìm kiếm, hệ thống mã hóa câu hỏi thành vector $1024$-dim và thực hiện tìm kiếm Láng giềng gần nhất (Nearest Neighbors) sử dụng **Cosine Similarity**. Kết quả trả về vượt trội so với Elasticsearch truyền thống vì tìm được cả những bài viết tương đồng về mặt lập luận logic (đồ thị tri thức) dù cách diễn đạt khác nhau.

### 2.6. Hệ thống Gợi ý Học thuật (Academic Recommender System)
* **Ứng dụng:** Đóng vai trò làm bộ máy gợi ý bài viết liên quan (Related papers recommendation).
* **Cơ chế:** Nếu một nhà nghiên cứu đang đọc bài báo $A$, hệ thống sẽ gợi ý bài báo $B$ có khoảng cách vector $v_{joint}$ ngắn nhất. Do vector Late Fusion lưu giữ cả ngữ cảnh văn bản và topo quan hệ thực thể khoa học, các bài báo được gợi ý sẽ có độ tương quan học thuật cực kỳ sâu sắc.

### 2.7. Trực quan hóa Không gian Tri thức (Knowledge Map Visualization)
* **Ứng dụng:** Trực quan hóa cách phân bổ của các lĩnh vực nghiên cứu khoa học để trình bày trước Hội đồng.
* **Cơ chế:** Áp dụng thuật toán giảm chiều phi tuyến **t-SNE** hoặc **UMAP** để ánh xạ không gian $1024$ chiều của tập Test xuống không gian $2$ chiều ($X, Y$).
* **Kết quả trực quan:** Vẽ Scatter Plot tô màu theo 21 lớp arXiv. Sự bổ trợ của cấu trúc GNN giúp các ngành học phân cụm rõ rệt, sắc nét, chứng minh trực quan sự thành công của thuật toán Late Fusion.
