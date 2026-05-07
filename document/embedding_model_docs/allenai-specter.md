**allenai/specter** là một *document-level embedding model* thuộc Allen Institute for AI (AllenAI), được thiết kế đặc biệt để biểu diễn ngữ nghĩa các **bài báo khoa học** phục vụ phân loại, truy xuất và gợi ý tài liệu.

**Đặc điểm kỹ thuật**
- Kiến trúc: SciBERT (BERT-based, pretrained trên corpus khoa học)
- Training type: Citation-informed Contrastive Learning
- Output: Vector 768 dimensions
- Tối ưu hóa cho document-level embeddings (toàn bài / abstract), không phải sentence-level.

**Vai trò trong paper**
Abstract → Trích xuất triples → Tạo 4 dạng biểu diễn → **Embedding** → Clustering/Classification

SPECTER là 1 trong 4 embedding model được so sánh, cùng với MiniLM, MPNet, và SciBERT.


*Clustering* => SPECTER đạt kết quả **trung bình** — tốt hơn SciBERT nhưng không vượt được MPNet và MiniLM

Với Abstract thuần, SPECTER đạt ARI ~0.415, NMI ~0.526 (so với MPNet ~0.470, MiniLM ~0.442)
SPECTER có **silhouette score cao nhất** (~0.095) → cluster hình học đẹp, tách biệt rõ trong không gian embedding
Nhưng ARI lại thấp hơn → cluster đó không khớp tốt với nhãn arXiv thực tế

=> INSIGHT: SPECTER có thể đang nhóm tài liệu theo đặc trưng từ *mạng trích dẫn* thay vì theo chủ đề phân loại. Geometry tốt ≠ alignment với ground truth tốt.


*Classification* => SPECTER **vượt trội** khi input là văn bản liên tục, phong phú

Với Abstract thuần: SPECTER đạt accuracy ~91.85% — tốt nhất trong số các model
Với Abstract+Triples (ghép nối): SPECTER đạt accuracy ~92.15% — vẫn dẫn đầu
Với Hybrid ([SEP] phân tách): SciBERT vượt lên (92.60%), SPECTER tụt lại (~91.8%)
Với Triples only: SciBERT tốt nhất (85.25%), SPECTER kém hơn rõ rệt

=> INSIGHT: SPECTER mạnh với *văn bản liên tục*, yếu với *input phân mảnh*. Pretraining trên citation network giúp nhận biết tốt ngữ cảnh nghiên cứu từ abstract đầy đủ, nhưng không generalize tốt sang dạng triples hay hybrid format.


**FINAL INSIGHT**: *Domain-specific pretraining* (trên corpus khoa học + citation graph) **không đảm bảo** hiệu suất clustering tốt hơn so với *semantic similarity pretraining* tổng quát. SPECTER thua MiniLM — model nhỏ hơn nhiều — trong clustering, vì MiniLM được tối ưu đúng mục đích: so sánh ngữ nghĩa giữa các văn bản.


**Vì sao SPECTER lại tốt cho classification nhưng kém cho clustering?**
=> SPECTER được train để nhận biết *sự liên quan giữa các bài báo* thông qua citation — phù hợp khi cần phân biệt các category rộng (classification).
=> Nhưng không gian embedding của nó phản ánh cấu trúc citation network, không nhất thiết tương đồng với taxonomy arXiv → cluster không khớp nhãn thực tế (ARI thấp).
=> Ngược lại, MiniLM/MPNet được finetune bằng **contrastive learning** trên semantic similarity → vector phân tách theo chủ đề tốt hơn khi clustering.

CƠ CHẾ HOẠT ĐỘNG
Anchor  →  Bài báo A (về NLP)
   │
   ├── Positive (A trích dẫn B) → Bài báo B (về NLP)
   │        ↑ KÉO LẠI GẦN NHAU
   │
   └── Negative (không có trích dẫn) → Bài báo C (về vật lý)
            ↑ ĐẨY RA XA NHAU

*Mục tiêu training:* Sau mỗi lần học, vector của Anchor và Positive (có quan hệ trích dẫn) phải gần nhau hơn, vector của Anchor và Negative (không liên quan) phải xa nhau hơn trong không gian embedding.
