**1.**
*Tên paper:* Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents.
*Chủ đề chính:* Áp dụng tri thức có cấu trúc để cải thiện việc phân loại và phân cụm tài liệu khoa học.

**2. Vấn đề paper giải quyết**
-   Số lượng bài báo khoa học đang tăng theo cấp số nhân (ví dụ: arXiv nhận tới 1.200 bài/ngày), khiến việc sắp xếp và phân loại thủ công trở nên bất khả thi. Các phương pháp tự động chỉ dựa trên văn bản thông thường (text-only) gặp khó khăn trong việc hiểu toàn bộ ngữ nghĩa của bài báo vì thuật ngữ chuyên ngành phức tạp và sự giao thoa giữa các lĩnh vực.

**3. Mục tiêu & Câu hỏi nghiên cứu**
-   Nghiên cứu này nhằm mục đích khám phá xem liệu việc đưa các tri thức có cấu trúc — cụ thể là các bộ ba *chủ thể-vị ngữ-khách thể (triples)* — vào biểu diễn văn bản có làm tăng hiệu quả của việc phân cụm (không giám sát) và phân loại (có giám sát) các bài báo khoa học hay không.

**4. Phương pháp nghiên cứu**
- Tác giả đề xuất một hệ thống pipelines sau:
*Trích xuất thông tin:* Sử dụng công cụ *spaCy* để trích xuất bộ 3 triplet *"chủ thể - vị ngữ - khách thể"* từ phần tóm tắt *(abstract)* của bài báo. 
    * Tạo 4 kiểu dữ liệu đầu vào: 
    (1) Chỉ dùng *Tóm tắt gốc*.
    (2) Chỉ dùng các *Triples*.
    (3) Nối liền *Tóm tắt + Triples.*
    (4) *Hybrid* (phân tách rõ ràng Tóm tắt và Triples).
*Nhúng văn bản (Embeddings):* Dùng 4 mô hình Transformer (MiniLM, MPNet, SciBERT, SPECTER) để vector hóa.
*Huấn luyện & Đánh giá:* Dùng KMeans, GMM, HDBSCAN cho tác vụ phân cụm, và dùng các mô hình Transformer cho tác vụ phân loại.

**5. Dữ liệu và Đối tượng nghiên cứu**
-   Link dataset: "https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download"
-   Dữ liệu được chia làm 2 phần tách biệt: 5.000 bài dùng cho thí nghiệm phân cụm và 10.000 bài dùng cho thí nghiệm phân loại.
-   Mỗi bài báo bao gồm phần tóm tắt văn bản đã được làm sạch và nhãn dán chủ đề của arXiv (ví dụ: cs.AI).


**6. Kết quả chính**
-   Trong việc *phân cụm (Clustering)*: Việc sử dụng *toàn bộ văn bản tóm tắt* mang lại các cụm mạch lạc nhất. Các mô hình nhẹ (MiniLM, MPNet) lại cho kết quả tốt hơn các mô hình chuyên về khoa học (SciBERT, SPECTER).
-   Trong việc *phân loại (Classification)*: Kiểu dữ liệu *Hybrid* (phân tách rõ tóm tắt và triples) mang lại kết quả cao nhất, đạt độ chính xác lên tới 92,6% và chỉ số macro-F1 là 0.925 khi chạy với mô hình SciBERT. Mô hình SciBERT cho thấy khả năng xuất sắc khi phân loại dữ liệu có cấu trúc.

**7. Đóng góp nổi bật của paper**
-   Nghiên cứu chứng minh được lợi ích của việc **kết hợp** *văn bản phi cấu trúc (chữ thông thường)* với *tri thức có cấu trúc (triples)*, mở ra cách thức tổ chức và phân loại tài liệu khoa học tốt hơn. Paper cũng xây dựng một quy trình làm việc (pipeline) dạng mô-đun chuẩn mực để các nghiên cứu sau có thể tái sử dụng.


**8. HẠN CHẾ của Paper**
-   Nếu chỉ dùng mỗi dữ liệu dạng triples, kết quả phân loại và phân cụm sẽ bị giảm sút do thiếu mất ngữ cảnh của câu văn.
-   Thuật toán phân cụm dựa trên mật độ (HDBSCAN) hoạt động rất kém vì không gian vector của văn bản nhiều chiều không có sự phân tách mật độ rõ ràng.


**9. Học được gì từ paper**
-   Mục 3.2 (Triples and Knowledge Graph Construction): Để hiểu cách họ dùng spaCy để trích xuất Chủ thể - Vị ngữ - Khách thể một cách tự động. Bạn hoàn toàn có thể code lại phần này cho dữ liệu tiếng Anh hoặc tiếng Việt.
-   Mục 3.3 (Text Representation Modes): Kỹ thuật tạo ra chuẩn đầu vào "Hybrid" sử dụng token [SEP] là một mẹo rất hay và dễ áp dụng (ví dụ: [Văn bản gốc] [SEP] [Các ý chính]).


