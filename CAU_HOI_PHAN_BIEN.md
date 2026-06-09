## CÂU HỎI
**Câu 0:** Output của bài báo
Bài báo đã chứng minh được 4 điểm cốt lõi:
- *Kết hợp tri thức có cấu trúc và phi cấu trúc* mang lại hiệu quả cao nhất: việc kết hợp văn bản tóm tắt tự nhiên (phi cấu trúc) với các bộ ba sự kiện chủ thể - vị ngữ - đối tượng (tri thức có cấu trúc) giúp tăng cường đáng kể khả năng phân loại tài liệu khoa học. Cụ thể, định dạng lai (Hybrid) với thẻ phân tách rõ ràng đã đẩy độ chính xác của bài toán phân loại lên mức cao nhất là 92.6%.
- *Mô hình phổ thông đánh bại mô hình chuyên ngành trong bài toán gom cụm:* Trái với dự đoán thông thường, bài báo chứng minh rằng các mô hình ngôn ngữ phổ thông nhưng được tối ưu hóa để so sánh độ tương đồng ngữ nghĩa (như MiniLM, MPNet) lại cho kết quả gom cụm (clustering) không giám sát vượt trội hơn hẳn so với các mô hình được huấn luyện chuyên sâu bằng khối lượng lớn dữ liệu khoa học (SciBERT, SPECTER)
- *Văn bản tự nhiên là nền tảng không thể vứt bỏ:* Mặc dù các bộ ba dữ kiện (triples) mang lại độ sắc bén để phân biệt các chuyên ngành hẹp, bài báo chứng minh rằng nếu chỉ dùng triples mà bỏ đi văn bản tóm tắt nguyên bản, hiệu suất gom cụm và phân loại sẽ giảm sút nghiêm trọng. Ngữ cảnh và các từ ngữ kết nối trong văn bản tự nhiên vẫn cung cấp tín hiệu cốt lõi nhất cho mô hình AI


**Câu 1:** Về định dạng dữ liệu và hiệu suất phân loại Bài báo đã thiết kế 4 định dạng đầu vào: Abstract, Triples, Abstract+Triples, và Hybrid. Bạn có thể giải thích tại sao định dạng Hybrid lại mang lại kết quả phân loại cao nhất (lên tới 92.6%), và điểm khác biệt cốt lõi trong cách cấu trúc của nó so với định dạng Abstract+Triples (chỉ nối trực tiếp) giúp mô hình hoạt động hiệu quả hơn là gì
*A:* Thẻ [SEP] cô lập các sự kiện (triples) khỏi đoạn văn xuôi (abstract) => Nhờ vậy mô hình có thể tập trung khai thác các dữ kiện có cấu cấu trúc mà không bị **nhiễu** bởi các câu chữ văn phong tự nhiên.

**Câu 2:** Về các thuật toán gom cụm (Clustering) Các tác giả đã thử nghiệm KMeans, GMM và HDBSCAN. Trong khi KMeans và GMM cho kết quả phân cụm bám sát với các chủ đề thực tế, HDBSCAN lại thất bại, chỉ tìm được rất ít cụm và gán một lượng lớn tài liệu là "nhiễu" (noise). Theo bài báo, nguyên nhân gốc rễ nào trong không gian vector (embedding space) đã làm khó một thuật toán dựa trên mật độ như HDBSCAN
*A:* Do **đặc tính hình học** của không gian vector đa chiều. HDBSCAN hoạt động bằng cách tìm kiếm các vùng dữ liệu dày đặc ngăn cách bởi các vùng trống rỗng. Bài báo giải thích rằng không gian embedding của các tài liệu không có *ranh giới phân tách mật độ rõ ràng (lack of clear density separations)*
=> Do đó thuật toán không thể tách các cụm và đành gán một lượng lớn điểm dữ liệu là *nhiễu*.

**Câu 3:** "Tại sao mô hình SciBERT dù được huấn luyện chuyên sâu bằng dữ liệu khoa học, nhưng lại cho kết quả gom cụm (clustering) kém xa các mô hình ngôn ngữ phổ thông như MiniLM hay MPNet?"
*A:* Mặc dù SciBERT được huấn luyện chuyên sâu bằng dữ liệu khoa học, có lợi thế về từ vựng chuyên ngành khoa học, nhưng nó không được tinh chỉnh đặc biệt cho việc đánh giá *độ tương đồng ngữ nghĩa (semantic similarity)*. Ngược lại *MPNET*, *MiniLM* ngay từ đầu đã được huấn luyện để tạo ra các vector đại diện câu (sentence embeddings) vô cùng chất lượng, tối ưu hóa riêng cho việc so sánh và gom cụm.

**Câu 4:** Sự "đổi ngôi" giữa các mô hình trong bài toán phân loại (Classification) Ở bài toán gom cụm, SciBERT hoạt động rất kém. Nhưng sang bài toán phân loại có giám sát, SciBERT lại tỏa sáng, đặc biệt là vượt trội hơn mô hình SPECTER khi đầu vào là định dạng Triples (chỉ có bộ ba) hoặc Hybrid. Trong khi đó, SPECTER lại làm tốt hơn khi đầu vào chỉ là Abstract. Bạn có thể giải thích vì sao hai mô hình này lại có thế mạnh ngược nhau với các loại dữ liệu đầu vào như vậy không?
*A:* SPECTER - mô hình này được huấn luyện trước trên *mạng lưới trích dẫn (citation graphs)* để tối ưu hóa việc hiểu toàn bộ tài liệu khoa học. Do đó, nó hoạt động cực kỳ hiệu quả khi có đủ ngữ cảnh văn xuôi liên tục (như khi đầu vào chỉ là Abstract). Tuy nhiên, SPECTER lại lúng túng khi gặp các đoạn dữ liệu cụt lủn hoặc bị phân mảnh.
Ngược lại, SciBERT được huấn luyện trên một tập dữ liệu khổng lồ gồm 1,14 triệu bài báo với bộ từ vựng chuyên ngành riêng (SciVocab), mang lại khả năng mô hình hóa ngôn ngữ cơ bản rất mạnh mẽ. Điều này giúp SciBERT dễ dàng thích nghi với văn phong "điện tín", ngắn gọn của định dạng Triples, cũng như xử lý trơn tru cấu trúc phân tách bằng thẻ [SEP] của định dạng Hybrid.

**Câu 5:** Khả năng "bóc tách" các chuyên ngành hẹp Khi hệ thống thử gom cụm các bài báo vật lý bằng cách chỉ dùng văn bản thuần túy (Abstract), các bài báo về "Vật lý vật chất ngưng tụ" (cond-mat) và "Vật lý lượng tử" (quant-ph) bị gộp chung vào một cụm. Nhưng khi dùng định dạng có chứa Triples (Abstract + Triples hoặc Hybrid), điều gì đã xảy ra với hai nhóm này? Kết quả đó chứng minh ưu điểm cụ thể gì của việc bổ sung Triples?
=> Việc bổ sung các *bộ ba tri thứ (Triples)* cung cấp thêm các *chi tiết sắc bén* giúp hệ thống nhận diện và bóc tách được các phân ngành nghiên cứu rất gần gũi nhau, điều mà văn xuôi thường dễ làm mờ đi.

**Câu 6:**Bí quyết "tuyến tính hóa" (Linearization) Như chúng ta đã nhắc sơ ở lần trước, Transformer không đọc thẳng cấu trúc đồ thị (như XML hay JSON). Sau khi hệ thống dùng công cụ NLP (spaCy) để trích xuất ra các bộ ba chủ thể - vị ngữ - đối tượng (ví dụ: transformer, improves, accuracy), tác giả đã phải làm một thao tác gọi là "linearization" (tuyến tính hóa) trước khi đưa vào mô hình. Bạn có nhớ thao tác này cụ thể là làm gì không?
=> Thay vì đưa thẳng cấu trúc đồ thị phức tạp vào mô hình, *linearization* là bước chuyển đổi các bộ ba dữ kiện (chủ thể, quan hệ, đối tượng) thành *những câu văn tự nhiên đơn giản hóa.*
Ví dụ cụ thể: Bộ ba (transformer, improves, accuracy) sẽ được chuyển thành câu hoàn chỉnh là "Transformer improves accuracy.". Thao tác này giúp giữ lại toàn bộ các khẳng định thực tế (factual assertions) ở một định dạng mà mô hình Transformer có thể đọc và mã hóa một cách dễ dàng.


**Câu 7:** Cách bài báo dùng *spacy* để trích xuất **Chủ thể - Vị ngữ - Khách thể**
- *Tiền xử lý văn bản:* Mỗi đoạn văn bản tóm tắt (abstract) trước tiên sẽ được *dàn phẳng - flatten* để xóa bỏ các dấu ngắt dòng.
- *Sử dụng pipeline chuyên khoa học:* Văn bản sau đó được xử lý qua một hệ thống spaCy đã được điều chỉnh (domain-adapted), được trang bị một mô hình ngôn ngữ chuyên biệt cho văn bản khoa học.
- *Xác định Vị Ngữ (Predicate)*
- *Định vị Chủ thể (Subject) và Khách thể (Object)*

**Câu 8:** Ở phần trích xuất Triples, các bạn dùng Spacy để phân tích cú pháp (dependency parsing). Tuy nhiên, ngôn ngữ khoa học trên ArXiv chứa rất nhiều thuật ngữ phức tạp hoặc công thức toán học. Các bạn đã làm sạch (tiền xử lý) như thế nào để đảm bảo Spacy trích xuất đúng Subject - Relation - Object?
=> Dùng mô hình chuyên dụng *spaCy sci model* chuyên tối ưu cho văn bản khoa học.

**Câu 9:** Các bạn có đưa ra một công thức tính điểm chất lượng Triple: S 
quality = 0.35Sdep +0.25Sphrase +0.25Srel +0.15S sentence (**Tổ hợp lồi**)
​Các trọng số (35%, 25%, 15%) này là do các bạn tự thực nghiệm rút ra hay tham khảo từ tài liệu nào? Nếu do nhóm tự đề xuất, cơ sở nào để cho rằng cấu trúc cú pháp (Dependency) lại quan trọng nhất (35%)?
=> Việc gán 35% cho Dependency là hợp lý bởi vì cấu trúc cú pháp đóng vai trò quyết định trong việc *định hình một mối quan hệ đúng đắn* giữa *Chủ ngữ* và *Vị ngữ.*

**Câu 10:** Nhóm sử dụng 3 thuật toán: KMeans, GMM và HDBSCAN. Trong slide có nói HDBSCAN có khả năng tự động gán nhãn nhiễu (-1) cho các tài liệu ngoại lai. Vậy khi bước sang giai đoạn Classification (Phase 4 & 5), tín hiệu của các cụm "nhiễu" này được xử lý hoặc lan truyền (Label Propagation) như thế nào vào mô hình phân loại?
Dùng K-Nearest Neighbors KNN (K=5, metric cosine) để tìm các tài liệu gần nhất trong không gian vector. Bỏ qua các láng giềng có nhãn nhiễu -1 nếu có ít nhất một láng giềng khác mang nhãn cụm hợp lệ. Trường hợp đều là nhiễu thì fallback phương án dự phòng.
Dịch ngữ nghĩa: Nếu nhãn lan truyền nhận được là -1, hệ thống tự động chuyển đổi nó thành chuỗi văn bản tự nhiên là "noise".
Tạo câu tín hiệu ngữ cảnh: Sinh ra chuỗi thông tin: "clustering mode [representation]. propagated cluster noise. confidence [độ tin cậy]."
Chèn tín hiệu ngữ nghĩa (Semantic Injection): Ghép chuỗi tín hiệu này trực tiếp vào văn bản đầu vào (ở đầu hoặc cuối đoạn văn) qua thẻ phân tách [SEP].
Huấn luyện mô hình: Mô hình Transformer (SciBERT/SPECTER) sẽ đọc tín hiệu "noise" dạng văn bản này để nhận biết đây là tài liệu ngoại lai/vùng biên, hỗ trợ việc phân loại nhãn chuyên ngành chính xác hơn.


**Câu 11:** Tại sao các bạn lại phải dùng Label Propagation từ bước Clustering để hỗ trợ Classification? Việc thêm trực tiếp nhãn cụm (cluster signal) vào chung với văn bản và triples có gây ra nhiễu thông tin nếu kết quả phân cụm trước đó bị sai không?
=> Mục đích của việc dùng Label Propagation là giúp *mô hình Classifier* **tận dụng thêm cấu trúc chủ đề** vĩ mô đã được phát hiện từ bước Clustering.
=> Chuỗi **"Abstract [SEP] Triples [SEP] Cluster Signal"** giúp mô hình Transformer học đồng thời cả ngữ nghĩa văn bản, quan hệ tri thức và cấu trúc chủ đề.
=> Tín hiệu phân cụm đóng vai trò như một thông tin bổ trợ chứ không phải nhãn quyết định.


**Câu 12:** Tại sao bạn lại chọn GNN?
=> Giải quyết điểm nghẽn giới hạn độ dài (max_length): Các mô hình như SciBERT hay SPECTER xử lý văn bản dưới dạng tuyến tính và bị giới hạn độ dài đầu vào (thường tối đa là 256 hoặc 512 tokens). Khi ghép nối cả Abstract và danh sách Triples dài, thông tin tri thức có cấu trúc ở phía sau sẽ bị cắt cụt (truncated), dẫn đến mất mát thông tin nghiêm trọng.
Bảo toàn cấu trúc topo đồ thị của Triples: Bản chất của các bộ ba tri thức (Subject - Predicate - Object) là một Đồ thị tri thức (Knowledge Graph). Nếu chúng ta chỉ đơn thuần "tuyến tính hóa" (linearization) chúng thành các câu văn thô để đưa vào Transformer, ta sẽ phá vỡ mất mối quan hệ topo đa chiều giữa các thực thể khoa học trong không gian. GNN ra đời để xử lý cấu trúc đồ thị bản sinh này.

[SEP]: token phân tách đặc biệt, tín hiệu tường minh giúp Transformer biết đâu là đâu là ngữ cảnh tự nhiên, đâu là cấu trúc tri thức bổ trợ.


**K-Means**
*1. ARI (Adjusted Rand Index - Chỉ số Rand hiệu chỉnh)*
*Ý nghĩa:* Đo lường mức độ tương đồng giữa kết quả phân cụm của KMeans và nhãn thực tế của dữ liệu dựa trên các cặp điểm dữ liệu.
*Cách hoạt động:* ARI sẽ đi quét toàn bộ dữ liệu và đếm:
Có bao nhiêu cặp tài liệu cùng lớp thực tế và được KMeans xếp cùng cụm.
Có bao nhiêu cặp tài liệu khác lớp thực tế và được KMeans xếp khác cụm.
Đặc điểm nổi bật (Adjusted): Chữ "Adjusted" có nghĩa là chỉ số này đã được hiệu chỉnh theo yếu tố ngẫu nhiên (chance).
Nếu bạn phân cụm bừa bãi (ngẫu nhiên), ARI sẽ bằng 0.
ARI = 1 → Kết quả gom cụm của KMeans trùng khớp hoàn hảo 100% với nhãn thực tế.
Thang đo: Từ -1 đến 1 (thông thường nằm trong khoảng 0 đến 1).
*2. NMI (Normalized Mutual Information - Thông tin tương hỗ chuẩn hóa)*
*Ý nghĩa:* Đo lường lượng thông tin dùng chung (chia sẻ) giữa kết quả phân cụm của KMeans và nhãn thực tế, dựa trên lý thuyết thông tin (Information Theory) và entropy.
*Cách hoạt động:* NMI trả lời câu hỏi: "Nếu tôi biết tài liệu này thuộc cụm số 5 của KMeans, tôi sẽ tự tin bao nhiêu % để đoán đúng nhãn thực tế của nó?"
Đặc điểm nổi bật (Normalized): Chỉ số này đã được chuẩn hóa về khoảng từ 0 đến 1 để bạn dễ dàng so sánh giữa các thuật toán có số lượng cụm khác nhau (ví dụ: so sánh KMeans với 20 cụm và HDBSCAN với 15 cụm).
NMI = 0 → Kết quả gom cụm hoàn toàn ngẫu nhiên, không chia sẻ tí thông tin nào với nhãn thực tế.
NMI = 1 → Kết quả gom cụm trùng khớp hoàn hảo với nhãn thực tế.
Chữ Normalized cực kỳ quan trọng. Nó giúp giải quyết một trò "gian lận" trong kiểm tra.

Nếu không có chuẩn hóa, một thuật toán "khôn lỏi" có thể gian lận bằng cách: Có 100 thẻ bài, nó chia luôn thành 100 chồng (mỗi chồng 1 thẻ). Lúc này, chắc chắn mỗi chồng đều "thuần khiết" 100% (vì làm gì có thẻ thứ hai mà lẫn lộn).

Hàm NMI có một cơ chế tự động phạt rất nặng những thuật toán chia quá nhiều cụm một cách vô tội vạ như vậy. Nhờ việc ép kết quả về thang đo chuẩn từ 0 đến 1, NMI giúp bạn so sánh sòng phẳng:

Thuật toán A chia thành 20 cụm được 0.7 điểm NMI.

Thuật toán B chia thành 5 cụm được 0.75 điểm NMI.



