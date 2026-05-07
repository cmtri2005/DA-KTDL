**all-MiniLM-L6-v2** là một *lightweight sentence embedding model* thuộc framework SentenceTransformers, được thiết kế đặc biệt cho các tác vụ *semantic similarity*

**Đặc điểm kỹ thuật**
- Kiến trúc: MiniLM (phiên bản nén của mô hình BERT)
- Training type: Constrative learning objective
- Output: Vector 384 dimensions
- Nhanh, nhẹ, effective với các tác vụ quy mô lớn.

**Vai trò trong paper**
Abstract → Trích xuất triples → Tạo 4 dạng biểu diễn → **Embedding** → Clustering/Classification

MiniLM là 1 trong 4 embedding model được so sánh, cùng với MPNet, SciBERT, và SPECTER.


*Clustering* => MiniLM chiếm ưu thế, vượt trội ở hầu hết mọi chỉ số - chỉ thua *MPNet* ở feature *Full Abstract*

Với Abstract thuần, MiniLM đạt ARI ~0.442 (á quân sau MPNet ~0.470)
Khi thêm triples vào → ARI của MiniLM tăng từ 0.442 lên 0.464 (hybrid)
Ngược lại, MPNet lại giảm từ 0.470 xuống ~0.40 khi thêm triples.

=> INSIGHT: MiniLM là model nhẹ hơn, nên hưởng lợi nhiều hơn từ structured knowledge bổ sung. MPNet vốn đã mạnh với text thuần nên thêm triples lại gây nhiễu.



**FINAL INSIGHT**: *Semantic similarity pretraining* quan trọng hơn *domain adaptation* cho unsupervised clustering. Dù SciBERT và SPECTER được train trên dữ liệu khoa học, nhưng vì không được fine-tune cho semantic similarity, chúng thua kém MiniLM — một model nhỏ hơn nhưng được tối ưu đúng mục đích.


**Vì sao MiniLM lại tốt cho clustering trong paper**
=> Vì BERT gốc và SciBERT được train để hiểu từng câu riêng lẻ, không tối ưu cho việc so sánh 2 văn bản với nhau
=> MiniLM được finetune thêm bằng **constrastive learning**.
Nhờ vậy, khi clustering, các vector của MiniLM phân tách rõ ràng theo chủ đề hơn — dù model nhỏ hơn SciBERT nhiều.

CƠ CHẾ HOẠT ĐỘNG
Anchor   →  "AI giúp phân loại văn bản khoa học"
   │
   ├── Positive (giống) → "Machine learning phân tích tài liệu"
   │        ↑ KÉO LẠI GẦN NHAU
   │
   └── Negative (khác)  → "Công thức nấu phở bò Hà Nội"
            ↑ ĐẨY RA XA NHAU

*Mục tiêu training:* Sau mỗi lần học, vector của Anchor và Positive phải gần nhau hơn, vector của Anchor và Negative phải xa nhau hơn trong không gian embedding.
