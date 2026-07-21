# PROGRESS NOTES — áp dụng legalrag_adjustments.md vào mã nguồn

Trạng thái: **Tất cả các mục hành động cụ thể (không cần nghiên cứu thêm) đã hoàn thành.**
Chỉ có 1 mục bị bỏ qua, và bị bỏ qua đúng theo chỉ định của chính tài liệu gốc (xem "Việc bỏ qua" bên dưới).

---

## 1. Việc đã làm (map theo đúng thứ tự ưu tiên §8 của legalrag_adjustments.md)

### Ưu tiên 1 — §2: Chuyển sang Qwen3.5-0.8B, tránh OOM/AttributeError
- **`backend/config.py`**: `GENERATION_MODEL_NAME` mặc định đổi sang `Qwen/Qwen3.5-0.8B-Instruct`; thêm `GENERATION_ENABLE_THINKING`, `GENERATION_ATTN_IMPL=sdpa`, `GENERATION_MAX_NEW_TOKENS_DEFAULT`; xoá `HYDE_MODEL_NAME` (chết, không còn dùng sau khi bỏ HyDE).
- **`backend/models.py`**: viết lại `_get_generation_model()` (bỏ `device_map`, thử `dtype=` rồi fallback `torch_dtype=`, `attn_implementation="sdpa"`, kiểm tra `chat_template is None` để fail rõ ràng) và `generate_text()` (tắt `enable_thinking` có fallback `TypeError`, cắt bỏ `<think>...</think>` nếu còn sót).

### Ưu tiên 2 — §3: Bỏ HyDE, thêm NER + Query Decomposition + Weighted RRF
- **`backend/retrieval/ner.py`** *(file mới)*: `extract_entities()`, `mask_person_org_entities()` dùng `NlpHUST/ner-vietnamese-electra-base`.
- **`backend/retrieval/querry_transform.py`**: xoá `generate_hyde()`, thêm `decompose_query()` (chỉ liệt kê khía cạnh pháp lý cần tra, không sinh văn bản luật giả định).
- **`backend/retrieval/hybrid_search.py`**: bỏ kênh HyDE; thêm masking tên riêng cho truy vấn luật, thêm kênh decomposition, đổi `_rrf_fuse` → `_rrf_fuse_weighted` (w_std=1.0 / w_agent=2.0 theo Judge-R1).

### Ưu tiên 3 — §5: Tách generation thành digest → verdict (giải quyết context ~10k token)
- **`backend/generation/case_digest.py`** *(file mới)*: `build_case_digest()` — tóm tắt top-N bằng chứng vụ án (N = `config.TOP_N_EVIDENCE_FOR_DIGEST`), fallback an toàn khi generation lỗi.
- **`backend/generation/prompt_builder.py`**: `build_prediction_prompt()` nay nhận `case_digest: str` thay vì `case_evidence_hits` + `compressed_texts`.
- **`backend/generation/generate.py`**: `predict_outcome()` nhận `case_digest` thay vì evidence thô; không còn gọi `compress_case_evidence`.
- **`backend/pipeline.py`**: `process_case()` build digest từ top-N evidence rồi mới gọi `predict_outcome`; **toàn bộ** `case_evidence_hits` vẫn được nộp vào `case_evidence` của submission (không bị cắt bớt) — chỉ đầu vào của LLM digest bị giới hạn.

### Ưu tiên 4 — §6: Fix wiring `build_parent_lookup` + soft-split đoạn quá dài
- **`backend/ingestion/chunker.py`**: thêm `_soft_split_oversized()` (cắt theo câu, ngưỡng `config.CHILD_MAX_CHARS=900`, không bao giờ cắt cứng theo token); tích hợp vào `chunk_article()`.
- **`backend/ingestion/parser.py`**: `load_law_corpus()` nay chấp nhận cả 2 dạng schema (`articles`/`text` **và** `content`/`content_Article` — dạng thứ hai là dạng `scripts/build_index.py` bản gốc thực sự đọc, nhưng `parser.py` trước đó không hỗ trợ). Đây là fix bắt buộc để có thể nối `build_index.py` qua `parser.py` → `chunker.py` như thiết kế gốc dự định.
- **`scripts/build_index.py`**: viết lại hoàn toàn — trước đây tự dựng `LawChunk` phẳng, bỏ qua `chunker.py`/`parser.py` (nên **không có** phân tách Khoản/Điểm thật, không có parent chunk). Nay dùng đúng `parser.load_law_corpus` → `chunker.chunk_articles` → `chunker.build_parent_lookup`, và **lưu parent lookup** ra `config.PARENT_LOOKUP_PATH` (`data/parent_lookup.pkl`).
- **`backend/pipeline.py`**: `collect_law_evidence()` load parent lookup, swap text sang parent (cả Điều) **chỉ để rerank**, sau rerank swap lại về child text (ngắn, đúng Khoản) trước khi đưa vào prompt cuối — đúng khuyến nghị "chỉ dùng parent-text cho bước rerank" trong §6.

### Ưu tiên 5 — §7: Retrieval-evaluator loop (tái dùng điểm rerank, không thêm LLM-judge)
- **`backend/pipeline.py`**: `collect_law_evidence()` — nếu điểm rerank cao nhất < `config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD` (mặc định 0.75, theo ViDRILL), tự động chạy thêm 1 vòng truy hồi dựa trên `decompose_query`, gộp candidate rồi rerank lại.

### Ưu tiên 6 — §4: Fix micro F1 trong test_all_backend.py + ước lượng E_i
- **`test/test_all_backend.py`**: giữ macro F1 cũ (đổi tên rõ `avg_f1` → log là "macro, approx"), **thêm micro F1 đúng công thức `evaluation.md §2.6`** (gộp TP/FP/FN toàn tập bằng `Counter` trước, chia sau); thêm ước lượng `E_i` trung bình theo `evaluation.md §2.4` dùng `n_segments` khi có, fallback `DEFAULT_MAX_API_CALLS_PER_CASE` khi không (khớp đúng fallback thật của `pipeline.py`).
- Ragas: **không thêm**, đúng kết luận của chính tài liệu gốc (kéo theo `langchain` + cần LLM-judge, đi ngược triết lý "không dùng framework RAG ngoài").

### Bonus (§7, mục "Neurosymbolic — rẻ, dễ implement", nằm trong ưu tiên 7 của §8)
- **`backend/generation/generate.py`**: sau khi lọc citation hallucination, nếu **không còn citation nào hợp lệ** mà model vẫn tự báo `confidence` cao, hạ trần `confidence` xuống ≤ 0.3 (không đổi nhãn dự đoán — việc đó cần phán đoán pháp lý theo case, ngoài phạm vi rule-based layer này).

### Hai mục nhỏ bổ sung sau khi rà lại lần 2 (đề cập ở §1, phần đối chiếu guideline.txt)
- **`backend/submission.py`**: thêm 1 dòng `log.warning` nhắc giới hạn 3 lần nộp/ngày trước khi chạy `main()`.
- **`test/test_all_backend.py`**: thêm comment rõ ràng rằng `verdict_label`/`related_law_provisions` **chỉ tồn tại ở Public Test**, không phải format Private Test chính thức (tránh nhầm lẫn).

### File cấu hình
- **`backend/.env.example`**: bổ sung toàn bộ biến môi trường mới (model, NER, decomposition, RRF weights, retrieval-evaluator, chunking, digest).

---

## 2. Việc chủ động **bỏ qua** (đúng theo chỉ định của legalrag_adjustments.md)

| Mục | Vì sao bỏ qua |
|---|---|
| **Late chunking (BambiBert)**, §7 | Chính tài liệu gốc kết luận: *"để giai đoạn R&D sau, không đưa vào core pipeline ngay"* — cần đội ngũ tự xác minh checkpoint trước khi đầu tư. Đây là mục 8 (cuối) trong thứ tự ưu tiên §8, được đánh dấu rõ là nghiên cứu sau. |
| **Semantic Chunking thuần** | Tài liệu gốc kết luận rõ ràng **"Không áp dụng"** — đi ngược nguyên tắc rule-based đã đặt ra vì rủi ro cắt ngang "trừ trường hợp". Không phải việc chưa làm, mà là việc **quyết định không làm**. |
| **`ragas`** trong pipeline chấm điểm chính thức | Tài liệu gốc kết luận rõ **"không nên thêm ragas vào pipeline chính"** (kéo theo langchain + cần LLM-judge, không khớp công thức `E_i` đặc thù của ALQAC). |

Không còn mục hành động cụ thể nào trong `legalrag_adjustments.md` bị bỏ sót.

---

## 3. Việc **CHƯA thể kiểm chứng** (do giới hạn môi trường, cần làm khi có hạ tầng thật)

Môi trường hiện tại không có GPU, không có Pinecone/HF Hub network access, nên các phần sau **mới chỉ được kiểm tra cú pháp + logic thuần Python** (py_compile, import graph, unit test có mock), **chưa chạy qua model thật**:

- `backend/models.py._get_generation_model()` / `generate_text()` — cần xác minh tên chính xác trên HF Hub của `Qwen/Qwen3.5-0.8B-Instruct` (tài liệu gốc tự lưu ý điều này) và test thật với GPU/CPU.
- `backend/retrieval/ner.py` — cần xác minh `NlpHUST/ner-vietnamese-electra-base` trả về đúng `entity_group` là `"PERSON"`/`"ORGANIZATION"` (label set thật của model, không chỉ suy đoán từ tên).
- `scripts/build_index.py` bản mới — cần chạy trên corpus JSON thật để xác nhận `load_law_corpus()` (đã hỗ trợ cả 2 schema) parse đúng 100% bản ghi, và số lượng parent/child chunk hợp lý.
- `backend/pipeline.py` — luồng end-to-end (retrieval-evaluator loop, swap parent/child text) mới test bằng logic giả lập (mock `RetrievedChunk`), chưa chạy với Pinecone/BM25/reranker thật.

Đã kiểm tra và **PASS**: cú pháp toàn bộ file (`py_compile`), toàn bộ import graph (`python -c "import backend...."`), logic soft-split + parent-lookup wiring (test thực tế với `RawArticle` giả lập), weighted RRF fusion, và luồng `predict_outcome` (citation hallucination drop + confidence cap) với `generate_text` được mock.

---

## 4. Cấu trúc file đã bàn giao

```
backend/
├── .env.example                  [SỬA] — biến môi trường mới
├── config.py                     [SỬA] — §2, §3, §5, §6, §7
├── models.py                     [SỬA] — §2 (loading an toàn cho model <1B)
├── pipeline.py                   [SỬA] — §5, §6, §7 (digest, parent-lookup, evaluator loop)
├── submission.py                 [SỬA] — nhắc giới hạn nộp bài
├── case_api_client.py            (không đổi)
├── generation/
│   ├── case_digest.py            [MỚI] — §5
│   ├── generate.py               [SỬA] — §5, §7 (neurosymbolic cap)
│   ├── prompt_builder.py         [SỬA] — §5
│   └── compress.py               (không đổi)
├── retrieval/
│   ├── ner.py                    [MỚI] — §3
│   ├── querry_transform.py       [SỬA] — §3 (bỏ HyDE, thêm decompose_query)
│   ├── hybrid_search.py          [SỬA] — §3 (weighted RRF, bỏ HyDE)
│   └── rerank.py                 (không đổi)
├── ingestion/
│   ├── chunker.py                [SỬA] — §6 (soft-split)
│   ├── parser.py                 [SỬA] — §6 (hỗ trợ 2 schema, tiền đề để build_index.py dùng được)
│   └── metadata.py               (không đổi)
└── indexing/                     (không đổi: embed.py, vector_store.py, bm25_index.py)

scripts/
└── build_index.py                [SỬA HOÀN TOÀN] — §6 (dùng đúng parser→chunker, lưu parent lookup)

test/
└── test_all_backend.py           [SỬA] — §4 (micro F1 + E_i), §1 (comment schema)

docs/submission_example.json, requirements.txt   (không đổi, đính kèm để tham chiếu)
```

**Không đổi / không cần sửa** (đã đúng theo `legalrag_adjustments.md`, hoặc không nằm trong phạm vi tài liệu):
`backend/case_api_client.py`, `backend/generation/compress.py`, `backend/retrieval/rerank.py`, `backend/ingestion/metadata.py`, `backend/indexing/*.py`, `backend/submission.py` (chỉ thêm 1 dòng log), `README.md`, `docs/*.md` khác (test_design.md, evaluation.md, case_content_api_doc.md, rag-system-design.md) — không nằm trong danh sách chỉnh sửa của adjustments.

---

## 5. Nếu cần tiếp tục (không có việc bắt buộc nào còn treo, nhưng nếu muốn đào sâu hơn)

Không có mục nào trong `legalrag_adjustments.md` còn ở trạng thái "chưa làm". Nếu muốn mở rộng thêm (ngoài phạm vi tài liệu gốc), các hướng tự nhiên tiếp theo là:
1. Chạy `scripts/build_index.py` với corpus thật + xác minh số chunk/parent hợp lý.
2. Chạy `test/test_all_backend.py --offline -n 5` (không cần `ALQAC_TOKEN`) để xác nhận pipeline không crash với model thật, rồi mới chạy full có `ALQAC_TOKEN`.
3. Tinh chỉnh `FINAL_LAW_TOP_K` (5 → 3–4) nếu vẫn thấy token cao ở bước verdict — đã để sẵn dạng biến môi trường, tài liệu gốc nêu đây là gợi ý tuỳ chỉnh thực nghiệm ("nếu vẫn thấy"), không phải thay đổi bắt buộc.
4. Late chunking (BambiBert) — nghiên cứu độc lập, ngoài phạm vi vòng nộp bài sắp tới (theo đúng khuyến nghị gốc).
