# Bộ Câu Hỏi Phản Biện Đồ Án (Mô Phỏng Giảng Viên)

> Mục tiêu: giúp bạn ôn đúng trọng tâm những gì **đã implement thực tế** trong repo hiện tại, và sẵn sàng trả lời cả câu hỏi kỹ thuật lẫn câu hỏi phản biện học thuật.

---

## 1) Tổng quan đề tài và phạm vi

### Câu 1. Đề tài của bạn giải quyết bài toán gì?
**Trả lời gợi ý:**
Đề tài tái hiện hướng làm trong paper về triples và knowledge-infused embeddings cho tài liệu khoa học arXiv. Mục tiêu là tạo pipeline từ tiền xử lý dữ liệu, trích triples, tạo nhiều representation văn bản, rồi đánh giá qua clustering/classification; sau đó làm ablation về triple quality ở Phase 7.


**Vì sao giảng viên hỏi câu này:**
Để kiểm tra bạn nắm “big picture” thay vì chỉ biết chạy code.

### Câu 2. Phần nào trong paper bạn đã implement thật sự, phần nào chưa?
**Trả lời gợi ý:**
Đã có pipeline Phase 1 (data + triples + 4 representation), các module cho embeddings/clustering/classification, báo cáo tái hiện kết quả, và extension Phase 7 về triple-quality-aware representations. Một số hướng nâng cao như learned quality scorer/graph-aware model mới ở mức đề xuất, chưa triển khai đầy đủ.

**Vì sao giảng viên hỏi câu này:**
Để tách rõ đóng góp thực nghiệm thật với phần định hướng tương lai.

### Câu 3. Tại sao bạn chọn arXiv metadata và cs.* làm dữ liệu chính?
**Trả lời gợi ý:**
Vì arXiv có quy mô lớn, đa nhãn lĩnh vực rõ, và phù hợp với mục tiêu phân loại tài liệu khoa học. Việc tập trung vào cs.* giúp bài toán nhất quán hơn và dễ tái lập split.

**Vì sao giảng viên hỏi câu này:**
Đánh giá tính hợp lý của dữ liệu so với bài toán.

---

## 2) Data pipeline (Phase 1)

### Câu 4. Pipeline Phase 1 tạo ra những output nào quan trọng nhất?
**Trả lời gợi ý:**
Hai file `cluster_combined.jsonl`, `classify_combined.jsonl` và các biến thể theo representation (`abstract`, `triples`, `concatenate`, `hybrid`) dạng JSONL/CSV. Mỗi record có `id`, `label`, triples, KG edges và text format tương ứng.

**Vì sao giảng viên hỏi câu này:**
Để xác minh bạn hiểu contract dữ liệu cho các phase sau.

### Câu 5. Bạn đảm bảo tính reproducible của split như thế nào?
**Trả lời gợi ý:**
Dùng random seed cố định (`seed=42`), tách rõ số mẫu clustering/classification (`5000/10000`), và có CLI tham số hóa để tái chạy cùng cấu hình.

**Vì sao giảng viên hỏi câu này:**
Reproducibility là tiêu chí bắt buộc khi báo cáo thực nghiệm.

### Câu 6. Bốn representation `abstract/triples/concatenate/hybrid` khác nhau thế nào?
**Trả lời gợi ý:**
- `abstract`: chỉ abstract gốc.
- `triples`: chỉ văn bản linearized từ triples.
- `concatenate`: ghép abstract + triples.
- `hybrid`: dạng kết hợp có phân tách rõ (`[SEP]`) để model nhận diện hai nguồn thông tin.

**Vì sao giảng viên hỏi câu này:**
Để kiểm tra bạn hiểu ý nghĩa ablation theo representation.

### Câu 7. Điểm yếu cố hữu của bước extract triples hiện tại là gì?
**Trả lời gợi ý:**
Phụ thuộc dependency parse nên có nhiễu cú pháp, rơi mất ngữ cảnh dài, và chất lượng triple bị ảnh hưởng bởi cấu trúc câu học thuật phức tạp.

**Vì sao giảng viên hỏi câu này:**
Đánh giá khả năng tự phản biện về chất lượng dữ liệu trung gian.

---

## 3) Thiết kế kỹ thuật và kiến trúc

### Câu 8. Vì sao bạn tách module theo `data_processing`, `embeddings`, `clustering`, `classification`, `phase7_quality`?
**Trả lời gợi ý:**
Để mỗi phase có trách nhiệm rõ ràng, dễ test độc lập, dễ thay thành phần (ví dụ đổi model embedding hoặc đổi scoring rule) mà không phá toàn pipeline.

**Vì sao giảng viên hỏi câu này:**
Kiểm tra tư duy thiết kế hệ thống, không chỉ thuật toán.

### Câu 9. Tại sao Phase 7 có 3 lệnh `build/classify/report`?
**Trả lời gợi ý:**
- `build`: sinh artifact chất lượng triple.
- `classify`: chạy ablation huấn luyện model.
- `report`: tổng hợp kết quả thành markdown/csv/figure.
Tách như vậy giúp chạy từng phần, debug nhanh, và tiết kiệm GPU.

**Vì sao giảng viên hỏi câu này:**
Đánh giá tính vận hành thực tế của pipeline.

### Câu 10. Nếu thiếu file đầu vào Phase 1/Phase 7 thì pipeline xử lý thế nào?
**Trả lời gợi ý:**
Code kiểm tra tồn tại file và ném lỗi rõ ràng (`FileNotFoundError`/`ValueError`) để người dùng biết cần chạy bước nào trước.

**Vì sao giảng viên hỏi câu này:**
Giảng viên muốn xem mức “production mindset” của đồ án.

---

## 4) Triple quality scoring (điểm nhấn phản biện)

### Câu 11. Công thức điểm chất lượng triple của bạn gồm những thành phần nào?
**Trả lời gợi ý:**
Điểm tổng là weighted sum của 4 thành phần:
- dependency rule score (35%)
- phrase length score (25%)
- relation frequency score (25%)
- source sentence score (15%)

**Vì sao giảng viên hỏi câu này:**
Đây là lõi phương pháp của Phase 7, chắc chắn sẽ bị hỏi sâu.

### Câu 12. Tại sao trọng số lại là 0.35/0.25/0.25/0.15?
**Trả lời gợi ý:**
Đây là heuristic ưu tiên độ tin cậy cú pháp (dependency rule) cao nhất, còn lại cân bằng giữa độ cô đọng phrase, độ đặc hiệu relation và độ sạch câu nguồn. Mình thừa nhận đây chưa phải tối ưu học được từ dữ liệu.

**Vì sao giảng viên hỏi câu này:**
Để xem bạn phân biệt được đâu là thiết kế heuristic, đâu là chứng minh thực nghiệm.

### Câu 13. `high_threshold` bạn chọn thế nào và vì sao chọn 0.75?
**Trả lời gợi ý:**
So sánh các ngưỡng cho thấy 0.65 quá lỏng, 0.85 quá gắt; 0.75 cho phân bố high/low cân bằng hơn (theo `triple_quality_summary.csv`), phù hợp để làm ablation.

**Vì sao giảng viên hỏi câu này:**
Ngưỡng là chỗ rất dễ bị phản biện là “chọn tùy ý”.

### Câu 14. `quality_hybrid_top5`, `top50`, `banded` khác nhau về ý tưởng gì?
**Trả lời gợi ý:**
- `top5`: giữ ít triple điểm cao nhất để giảm nhiễu mạnh.
- `top50`: giữ nửa trên theo chất lượng để cân bằng thông tin.
- `banded`: giữ cả high và low nhưng phân tách band, giúp model tự học mức quan trọng.

**Vì sao giảng viên hỏi câu này:**
Để đánh giá bạn có thiết kế ablation có chủ đích hay không.

### Câu 15. Hạn chế của scoring rule-based hiện tại là gì?
**Trả lời gợi ý:**
Chưa nắm được semantic utility thực sự của triple theo downstream task; có thể loại nhầm triple hữu ích hoặc giữ triple “đẹp cú pháp nhưng ít thông tin”.

**Vì sao giảng viên hỏi câu này:**
Câu then chốt để bạn chứng minh tư duy nghiên cứu chín chắn.

---

## 5) Huấn luyện và đánh giá

### Câu 16. Trong run Phase 7 của bạn, cấu hình train chính là gì?
**Trả lời gợi ý:**
SciBERT (`allenai/scibert_scivocab_uncased`), 10k mẫu classify, split 8k/2k, best epoch = 3, so sánh giữa `hybrid`, `quality_hybrid_top5`, `quality_hybrid_banded`.

**Vì sao giảng viên hỏi câu này:**
Để xác định bạn nhớ đúng bối cảnh khi diễn giải số liệu.

### Câu 17. Kết quả chính của Phase 7 là gì?
**Trả lời gợi ý:**
Trong cùng run, baseline `hybrid` tốt nhất: accuracy 0.8235, macro-F1 0.5688. Hai biến thể quality đều giảm nhẹ, nên filtering rule-based chưa cải thiện.

**Vì sao giảng viên hỏi câu này:**
Đây là “takeaway” phải trả lời dứt khoát.

### Câu 18. Vì sao accuracy có thể gần nhau nhưng macro-F1 lại giảm đáng chú ý?
**Trả lời gợi ý:**
Macro-F1 trung bình đều theo nhãn nên nhạy với lớp nhỏ. Một vài nhãn nhỏ giảm mạnh (ví dụ `nlin`) kéo macro-F1 xuống dù accuracy tổng thể giảm ít.

**Vì sao giảng viên hỏi câu này:**
Để kiểm tra bạn hiểu ý nghĩa metric chứ không chỉ đọc số.

### Câu 19. Tại sao không kết luận trực tiếp “Phase 7 tốt hơn/g kém hơn Phase 5”?
**Trả lời gợi ý:**
Vì setup khác nhau (Phase 5 có propagation signal, pipeline tuning khác). So sánh công bằng nhất là trong cùng điều kiện Phase 7 giữa các representation.

**Vì sao giảng viên hỏi câu này:**
Kiểm tra tính nghiêm túc về thiết kế thực nghiệm và so sánh.

### Câu 20. Top-3 accuracy cao nói lên điều gì?
**Trả lời gợi ý:**
Model thường đưa nhãn đúng vào top dự đoán dù top-1 chưa tối ưu, phù hợp bối cảnh nhãn khoa học có chồng lấn ngữ nghĩa.

**Vì sao giảng viên hỏi câu này:**
Giảng viên muốn bạn đọc sâu hơn ngoài accuracy top-1.

---

## 6) Câu hỏi “gài” thường gặp khi bảo vệ

### Câu 21. Kết quả âm (quality filtering không tăng điểm) có còn giá trị khoa học không?
**Trả lời gợi ý:**
Có. Nó bác bỏ giả thuyết đơn giản rằng “lọc heuristic là đủ”, từ đó định hướng rõ bước tiếp theo (learned scorer, better extractor, graph-aware modeling).

**Vì sao giảng viên hỏi câu này:**
Để đánh giá tư duy nghiên cứu thay vì “chỉ thích kết quả đẹp”.

### Câu 22. Nếu giảng viên yêu cầu chứng minh pipeline không bị data leakage, bạn trả lời gì?
**Trả lời gợi ý:**
Nêu rõ split độc lập theo seed trước các bước train; dữ liệu classify dùng file riêng; trong classify chỉ dùng text/label của split đó; không dùng thông tin nhãn validation/test khi build feature.

**Vì sao giảng viên hỏi câu này:**
Leakage là lỗi phổ biến khiến điểm số “ảo”.

### Câu 23. Vì sao không dùng luôn mô hình end-to-end trích triple + phân loại?
**Trả lời gợi ý:**
Phạm vi đồ án ưu tiên tái hiện có kiểm soát và phân tích thành phần. Tách pipeline giúp quan sát đóng góp từng khối rõ hơn, thuận lợi cho ablation.

**Vì sao giảng viên hỏi câu này:**
Để xem bạn kiểm soát scope có hợp lý không.

### Câu 24. Nếu tăng dữ liệu hoặc đổi domain ngoài arXiv, bạn kỳ vọng điều gì thay đổi?
**Trả lời gợi ý:**
Relation frequency và noise profile sẽ đổi đáng kể; threshold/rule có thể không còn tối ưu, cần hiệu chỉnh lại hoặc học scorer theo domain.

**Vì sao giảng viên hỏi câu này:**
Kiểm tra khả năng khái quát hóa và nhận diện giới hạn mô hình.

---

## 7) Câu hỏi về cải tiến tương lai (nên chuẩn bị trước)

### Câu 25. Bước cải tiến ưu tiên số 1 nếu có thêm thời gian là gì?
**Trả lời gợi ý:**
Thay rule-based quality bằng learned scorer (supervised/weak supervision) để quality phản ánh utility cho downstream classification tốt hơn.

**Vì sao giảng viên hỏi câu này:**
Để đánh giá bạn biết “nút thắt cổ chai” thật sự nằm ở đâu.

### Câu 26. Bạn sẽ thiết kế thí nghiệm nào để kiểm chứng cải tiến đó?
**Trả lời gợi ý:**
Giữ nguyên backbone classifier và split, chỉ thay module scoring; chạy cùng bộ representation + seed; báo cáo delta accuracy/macro-F1/per-label và kiểm định ổn định qua nhiều seed.

**Vì sao giảng viên hỏi câu này:**
Kiểm tra năng lực thiết kế experiment công bằng.

### Câu 27. Có hướng nào không cần bỏ triple nhưng vẫn giảm nhiễu?
**Trả lời gợi ý:**
Có: gating/attention theo triple score thay vì hard filtering, hoặc thêm special tokens để model học trọng số high-vs-low mềm dẻo hơn.

**Vì sao giảng viên hỏi câu này:**
Để xem bạn có tư duy “soft decision” thay cho cắt ngưỡng cứng.

### Câu 28. Bạn sẽ bổ sung kiểm thử gì để pipeline đáng tin cậy hơn?
**Trả lời gợi ý:**
Thêm unit tests cho scoring edge-cases, test contract schema giữa phase, regression tests cho metric outputs, và smoke test CI cho CLI `build/classify/report` với `--limit`.

**Vì sao giảng viên hỏi câu này:**
Đánh giá mức trưởng thành kỹ thuật của đồ án.

---

## 8) Mẫu trả lời nhanh khi bị hỏi dồn

### Câu 29. “Một câu chốt đóng góp chính của đồ án?”
**Trả lời gợi ý:**
Đồ án xây được pipeline tái hiện end-to-end có thể chạy lại, và cung cấp bằng chứng thực nghiệm rằng triple-quality filtering kiểu heuristic chưa vượt baseline hybrid trong thiết lập hiện tại.

**Vì sao giảng viên hỏi câu này:**
Để buộc bạn cô đọng giá trị cốt lõi.

### Câu 30. “Nếu bảo vệ lại, bạn đổi 1 thứ duy nhất?”
**Trả lời gợi ý:**
Mình sẽ ưu tiên learned quality scorer và chạy đủ các representation còn thiếu (đặc biệt `quality_hybrid_top50`) để kết luận Phase 7 toàn diện hơn.

**Vì sao giảng viên hỏi câu này:**
Để đánh giá mức tự học và khả năng cải tiến sau kết quả hiện tại.

---

## Gợi ý cách ôn trong 30-45 phút trước khi báo cáo

1. Học thuộc 5 câu “xương sống”: 1, 4, 11, 17, 29.
2. Chuẩn bị số liệu cụ thể để nói trơn tru: `0.8235`, `0.5688`, ngưỡng `0.75`, split `5000/10000`.
3. Tập trả lời 3 câu phản biện khó: 13, 19, 21.
4. Chốt sẵn 1 roadmap cải tiến ngắn: learned scorer + thêm ablation + multi-seed.

