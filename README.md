# arXiv Triples Phase 1

Repo hiện tại triển khai **Phase 1** của paper *Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents*.

Phase này dùng để chuẩn hóa dữ liệu đầu vào cho các phase sau:

- đọc metadata arXiv từ file JSONL
- lọc, làm sạch và chia dữ liệu thành 2 split không overlap
- trích xuất triples `(subject, relation, object)` từ abstract bằng spaCy/scispaCy
- lưu triples dưới dạng text và document-level knowledge graph edges
- tạo 4 dạng biểu diễn văn bản:
  - `Abstract`
  - `Triples`
  - `Concatenate` / `Abstract+Triples`
  - `Hybrid`
- ghi output JSONL/CSV để dùng cho embedding, clustering và classification

## Trạng thái hiện tại

- Đã implement: Phase 1 data processing.
- Chưa implement trong code: embedding generation, clustering, classification, và hướng NCKH mở rộng.
- Lộ trình các phase tiếp theo nằm ở [paper_reimplementation_todo.md](paper_reimplementation_todo.md).

## Yêu cầu

- Python `3.10+`
- file dataset arXiv metadata JSONL
- đủ dung lượng đĩa để lưu output

Dataset mặc định:

```text
dataset/arxiv-metadata-oai-snapshot.json
```

## Cấu trúc repo liên quan

```text
arxiv_triples_pipeline.py       # entrypoint chạy Phase 1
requirements.txt               # Python dependencies cơ bản
paper_reimplementation_todo.md  # roadmap reproduce paper + hướng phát triển
data_processing/
  cli.py                       # parse CLI args
  constants.py                 # hằng số dùng chung
  loading.py                   # load/filter/dedup/split dữ liệu
  triples.py                   # load spaCy model + extract triples/KG edges
  representations.py           # build 4 representations
  output.py                    # ghi JSONL/CSV
  pipeline.py                  # orchestration chính
dataset/
  arxiv-metadata-oai-snapshot.json
```

## Cài đặt

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

### Linux / macOS / WSL

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

Kiểm tra model đã cài thành công:

```bash
python3 -c "import spacy; spacy.load('en_core_sci_md'); print('ok')"
```

## Cách chạy

Nên chạy từ root repo bằng module entrypoint:

```bash
python3 -m arxiv_triples_pipeline
```

Lệnh trên chạy với default:

- input: `dataset/arxiv-metadata-oai-snapshot.json`
- output: `outputs/phase1_data`
- clustering split: `5000`
- classification split: `10000`
- spaCy model: `en_core_sci_md`
- seed: `42`

Paper chỉ nói dùng subset "recent years" nhưng không công bố chính xác mốc năm. Nếu muốn tái lập có kiểm soát hơn, nên ghi rõ năm lọc khi chạy.

### Smoke test

```bash
python3 -m arxiv_triples_pipeline \
  --output outputs/phase1_data_smoke \
  --n_cluster 100 \
  --n_classify 200 \
  --spacy_model en_core_sci_md \
  --year_from 2019 \
  --year_to 2025 \
  --batch_size 128 \
  --seed 42
```

### Full run

```bash
python3 -m arxiv_triples_pipeline \
  --output outputs/phase1_data \
  --n_cluster 5000 \
  --n_classify 10000 \
  --spacy_model en_core_sci_md \
  --year_from 2019 \
  --year_to 2025 \
  --batch_size 256 \
  --seed 42
```

Nếu muốn dùng toàn bộ dataset đủ điều kiện, bỏ `--year_from` và `--year_to`. Pipeline sẽ cảnh báo vì lựa chọn này không ghi rõ mốc "recent years".

## Tham số chính

- `--input`: đường dẫn tới file arXiv metadata JSONL
- `--output`: thư mục output
- `--n_cluster`: số document cho split clustering
- `--n_classify`: số document cho split classification
- `--spacy_model`: spaCy/scispaCy model dùng để trích triples
- `--year_from`: năm bắt đầu để lọc dữ liệu
- `--year_to`: năm kết thúc để lọc dữ liệu
- `--batch_size`: batch size cho `nlp.pipe`
- `--seed`: random seed

## Output sinh ra

Trong thư mục output, pipeline tạo:

```text
cluster_combined.jsonl
classify_combined.jsonl
cluster_abstract.jsonl / .csv
cluster_triples.jsonl / .csv
cluster_concatenate.jsonl / .csv
cluster_hybrid.jsonl / .csv
classify_abstract.jsonl / .csv
classify_triples.jsonl / .csv
classify_concatenate.jsonl / .csv
classify_hybrid.jsonl / .csv
```

`*_combined.jsonl` chứa record đầy đủ:

- `id`, `title`, `categories`, `primary_category`, `label`, `update_date`
- `n_triples`
- `triples`
- `kg_edges`
- `triples_text`
- `fmt_abstract`
- `fmt_triples`
- `fmt_concatenate`
- `fmt_hybrid`

Các file theo từng representation có cột gọn hơn:

```text
id, label, primary_category, n_triples, text
```

## Kiểm tra sau khi chạy

- Kiểm tra đủ 18 file output.
- Mở `cluster_combined.jsonl` hoặc `classify_combined.jsonl` để inspect vài dòng.
- Xem `n_triples` trung bình có hợp lý không.
- Kiểm tra thủ công vài `triples` và `kg_edges` để đánh giá nhiễu từ dependency parsing.

## Best practice

- Chạy smoke test trước full run.
- Dùng virtual environment riêng cho repo.
- Ghi rõ `--year_from` và `--year_to` khi cần reproduction nhất quán.
- Không ghi đè output cũ nếu đang so sánh nhiều lần thực nghiệm.
- Giữ `--seed 42` để split ổn định.
- Nếu đổi `spacy_model`, xem đó là thay đổi thực nghiệm vì triples sinh ra có thể khác.
