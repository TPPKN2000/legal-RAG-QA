# Kiến trúc RAG cho Legal QA (Self-hosted, không dùng framework)

> Mục tiêu: hệ thống hỏi-đáp pháp luật tiếng Việt, tự host toàn bộ (embedding + LLM ≤10B tham số), không phụ thuộc LangChain/LlamaIndex, code tự viết để kiểm soát chặt từng bước — quan trọng với domain pháp lý vì sai sót có hậu quả nghiêm trọng.

---

## 1. Sơ đồ tổng thể

```
Câu hỏi người dùng
        │
        ▼
┌───────────────────────────────────────────┐
│ 1. PRE-RETRIEVAL                            │
│  • Query Rewriting (3-5 biến thể)           │
│  • HyDE (câu trả lời giả định)              │
│  • Hierarchical Chunking (Child ↔ Parent)   │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ 2. RETRIEVAL                                │
│  • Hybrid Search (BM25 + Vector)            │
│  • Metadata Filtering (hiệu lực, loại VB)   │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ 3. POST-RETRIEVAL                           │
│  • Reranking (Cross-Encoder)                │
│  • Prompt Compression (chỉ nén phần phụ)    │
└───────────────────────────────────────────┘
        │
        ▼
LLM sinh câu trả lời + trích dẫn điều khoản
```

---

## 2. Giai đoạn Pre-retrieval

### 2.1 Query Rewriting / Expansion
- Dùng LLM tự host sinh 3-5 câu hỏi tương đương/chi tiết hơn.
- **Riêng cho pháp lý**: cần thêm bước chuẩn hóa thuật ngữ (ngôn ngữ đời thường → thuật ngữ luật), lý tưởng kết hợp một từ điển đồng nghĩa pháp lý (ontology) thay vì chỉ dựa LLM.

### 2.2 Hierarchical Chunking
- Child chunk nhỏ (~200 token) chứa chi tiết, liên kết với Parent chunk lớn (~1000 token) chứa ngữ cảnh đầy đủ.
- **Riêng cho pháp lý**: tận dụng cấu trúc gốc Chương > Mục > Điều > Khoản > Điểm làm ranh giới chunk (rule-based, không cắt token cứng).
  - Parent chunk = toàn bộ Điều luật; Child chunk = từng Khoản/Điểm.
  - Mỗi chunk giữ metadata "breadcrumb" phân cấp để trích dẫn chính xác.
  - Xử lý tham chiếu chéo giữa các điều khoản (enrichment hoặc knowledge graph).

### 2.3 HyDE (Hypothetical Document Embeddings)
- LLM sinh câu trả lời giả định → dùng chính câu đó để tìm kiếm vector (text-đối-text khớp hơn câu hỏi-đối-text).
- **Rủi ro cần lưu ý**: LLM có thể "bịa" nội dung nghe giống văn phong luật nhưng sai — nên kết hợp với Hybrid Search để giảm lệch hướng, và log lại để debug.

---

## 3. Giai đoạn Retrieval

### 3.1 Hybrid Search (BM25 + Vector)
- BM25: chính xác tuyệt đối về từ khóa, số hiệu văn bản, mã điều luật (VD: "Nghị định 145/2020/NĐ-CP").
- Vector: hiểu ngữ nghĩa, từ đồng nghĩa.
- Merge 2 danh sách bằng **Reciprocal Rank Fusion (RRF)** thay vì cộng điểm trực tiếp (thang điểm BM25 và cosine similarity không tương đồng).

### 3.2 Metadata Filtering
- Lọc cứng trước khi tìm vector để tránh tìm kiếm lan man.
- **Bắt buộc với pháp lý**: lọc theo hiệu lực thời gian (`effective_date`, `expiry_date`) — luật hay bị sửa đổi/thay thế, nếu không lọc có thể trích dẫn văn bản đã hết hiệu lực.
- Metadata tối thiểu: loại văn bản, cơ quan ban hành, số hiệu, ngày ban hành, ngày hiệu lực, trạng thái (còn hiệu lực/hết hiệu lực/sửa đổi), văn bản thay thế/được thay thế.

---

## 4. Giai đoạn Post-retrieval

### 4.1 Reranking
- Sau khi có top 20-30 kết quả, dùng Cross-Encoder chấm điểm lại theo độ tương quan.
- Model gợi ý: **AITeamVN/Vietnamese_Reranker** (chuyên tiếng Việt) hoặc **BAAI/bge-reranker-v2-m3** (đa ngôn ngữ, dự phòng).

### 4.2 Prompt Compression
- Dùng LLMLingua loại bỏ từ thừa để tiết kiệm token.
- **Lưu ý riêng pháp lý**: KHÔNG nén phần nội dung điều luật gốc (giữ nguyên văn) — mất một từ nối như "trừ trường hợp", "ngoại trừ" có thể đảo ngược ý nghĩa điều khoản. Chỉ nén phần ngữ cảnh phụ/kết quả thừa.

---

## 5. Model tự host (≤10B tham số)

| Vai trò | Model đề xuất | Ghi chú |
|---|---|---|
| Embedding | **AITeamVN/Vietnamese_Embedding** (v1/v2, ~560M) | Fine-tune từ BGE-M3, huấn luyện trên 300K–1,1M triplet câu hỏi–tài liệu tiếng Việt |
| Embedding (thử nghiệm) | **truro7/vn-law-embedding** | Huấn luyện chuyên biệt cho legal QA tiếng Việt, dùng Matryoshka loss, nên A/B test với model trên |
| Reranker | **AITeamVN/Vietnamese_Reranker** | Cùng series với embedding, tương thích tốt |
| Reranker (dự phòng) | **BAAI/bge-reranker-v2-m3** | Đa ngôn ngữ, ổn định |
| LLM sinh câu trả lời | **Qwen3-8B** | License Apache 2.0, hỗ trợ tiếng Việt tốt, có chế độ reasoning bật/tắt |
| LLM (nhẹ hơn) | **Qwen3-4B** | Khi hạ tầng hạn chế, đánh đổi chất lượng suy luận |

**Khuyến nghị bổ sung**: LoRA fine-tune nhẹ Qwen3-8B trên vài nghìn cặp (câu hỏi, context, câu trả lời chuẩn có trích dẫn) để cải thiện tính nhất quán khi trích dẫn và khả năng từ chối trả lời khi thiếu căn cứ.

### Hạ tầng serving
- **vLLM** — throughput tốt cho concurrent request, hỗ trợ tốt Qwen3.
- **Ollama** — đơn giản cho dev/scale nhỏ.
- Embedding model (560M) có thể chạy CPU nếu traffic thấp, hoặc share GPU nhỏ với LLM (quantize INT4/INT8).

---

## 6. Tech stack (không dùng framework RAG)

| Chức năng | Thư viện |
|---|---|
| Parse PDF/docx | `pdfplumber`, `python-docx` |
| Đếm token | `tiktoken` |
| Embedding | `sentence-transformers` |
| Vector DB | `qdrant-client`, `chromadb`, hoặc `faiss` |
| BM25 | `rank_bm25` (nhỏ) hoặc OpenSearch/Elasticsearch (production) |
| Cross-encoder rerank | `sentence-transformers.CrossEncoder` |
| Nén prompt | `llmlingua` |
| LLM serving | `vllm` hoặc `ollama` |
| Data schema | `pydantic` |

### Cấu trúc thư mục
```
legal-rag/
├── ingestion/      # parser, chunker, metadata
├── indexing/       # embed, vector_store, bm25_index
├── retrieval/      # query_transform, hybrid_search, rerank
├── generation/      # compress, prompt_builder, generate
├── models.py        # Pydantic schemas dùng chung
└── pipeline.py       # orchestration chính
```

---

## 7. Các phần bắt buộc thêm ngoài mô tả gốc

1. **Grounding & Citation nghiêm ngặt** — LLM buộc trích dẫn Điều/Khoản/Điểm cho mọi luận điểm, từ chối trả lời nếu context không đủ căn cứ. Có bước hậu kiểm (verification pass) so khớp câu trả lời với chunk đã retrieve.
2. **Data ingestion pipeline riêng** — parser tách cấu trúc Điều/Khoản/Điểm, tự động gắn metadata, phát hiện quan hệ sửa đổi/bãi bỏ giữa văn bản, versioning ở tầng document store.
3. **Evaluation** — test set câu hỏi pháp lý có đáp án chuẩn (chuyên gia soát), đo Recall@k, MRR cho retrieval và faithfulness/citation accuracy cho câu trả lời.
4. **Guardrail & disclaimer** — luôn ghi rõ đây là công cụ hỗ trợ tra cứu, không thay thế tư vấn pháp lý chính thức.

---

## 8. Thứ tự triển khai đề xuất

1. **Tuần 1**: Ingestion + chunking theo cấu trúc điều luật (ảnh hưởng chất lượng toàn hệ thống nhất).
2. **Tuần 2**: Indexing (vector + BM25) + metadata filtering theo hiệu lực — đo Recall@10.
3. **Tuần 3**: Hybrid search + reranking — xác nhận cải thiện Recall@10 trước khi thêm phức tạp.
4. **Tuần 4**: Query rewriting/HyDE — chỉ thêm sau khi retrieval nền tảng ổn định.
5. **Cuối**: Prompt compression + grounding verification (lớp tối ưu, không phải lớp nền).

> Nguyên tắc: build từng module rời rạc, đo bằng eval set thật trước rồi mới ghép — tránh viết hết pipeline một lượt rồi mới test, vì rất khó debug lỗi nằm ở khâu nào nếu build "big bang".
