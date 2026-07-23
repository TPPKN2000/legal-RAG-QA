# LegalRAG — Đề xuất điều chỉnh

---

## 0. Tóm tắt phát hiện quan trọng nhất (đọc trước)

1. **Private Test chỉ có `case_id` + `case_query`, KHÔNG có `n_segments`.** → `_case_api_budget()` trong `pipeline.py` sẽ *luôn* rơi vào nhánh `DEFAULT_MAX_API_CALLS_PER_CASE` khi chấm thật, chứ không phải nhánh `2*n_i`. Đây là thứ cần tối ưu kỹ nhất vì nó quyết định toàn bộ hệ số phạt `E_i`.
2. **`build_parent_lookup()` trong `chunker.py` được viết ra nhưng không hề được gọi ở đâu trong `pipeline.py`/`rerank.py`.** Reranker hiện đang chấm điểm trên *child chunk* (một Khoản/Điểm ngắn) thay vì *parent chunk* (cả Điều) — trong khi paper ViDRILL (đính kèm) chỉ ra rerank trên đoạn dài cho kết quả tốt hơn.
3. **`test_all_backend.py` tính F1 theo kiểu macro (trung bình per-case)**, trong khi `evaluation.md §2.6` yêu cầu **micro F1** (gộp TP/FP/FN toàn tập trước rồi mới chia). Hai con số này có thể lệch đáng kể.
4. **Context ~10k token không đến từ HyDE hay query rewriting** (các bước này chỉ dùng case_query ngắn) mà đến từ **prompt cuối cùng** trong `prompt_builder.py`, nơi toàn bộ *law text verbatim* + *case evidence* được nhét chung vào 1 lần gọi LLM.
5. Model `Qwen/Qwen3-8B` hiện tại **chắc chắn OOM trên T4 15GB** (8B tham số × 2 byte fp16 ≈ 16GB chỉ riêng trọng số, chưa kể activation + 2 model phụ). Chuyển sang Qwen3.5-0.8B là bắt buộc, không phải tùy chọn.

Phần dưới đây trình bày chi tiết từng mục theo đúng thứ tự yêu cầu.

---

## 1. Đối chiếu repo ↔ `pipeline.txt` ↔ `guideline.txt`

| Bước trong `pipeline.txt` | Repo hiện tại (`backend/`) | Nhận xét |
|---|---|---|
| 1. Lưu corpus vào Pinecone (VNBertLaw) + **Semantic Chunking** | `vector_store.py` dùng Pinecone + `AITeamVN/Vietnamese_Embedding`; chunking là **rule-based Điều/Khoản/Điểm** (`chunker.py`), không phải semantic chunking | Cách làm của repo **an toàn hơn** cho domain pháp lý (đúng như `design_doc §2.2` tự cảnh báo semantic chunking có thể cắt ngang "trừ trường hợp"). Giữ nguyên hướng rule-based, chỉ bổ sung dual-granularity (mục 6). |
| 2. Query qua NER (`NlpHUST/ner-vietnamese-electra-base`) | **Không có bước NER nào trong repo** | Thiếu hoàn toàn — cần bổ sung (mục 3, 8). |
| 3. Gọi Case Content API | `case_api_client.py` + `pipeline.collect_case_evidence` | Đã đúng tinh thần, rate-limit 5s/req khớp `case_content_api_doc.md`. |
| 4. HyDE bằng "Qwen Law VLSP" | `querry_transform.generate_hyde()` dùng model sinh chung (`GENERATION_MODEL_NAME`) | Cần **bỏ** theo yêu cầu (mục 3). |
| 5. Hybrid Search BM25 + Vector, **alpha = 0.7** | `hybrid_search.py` dùng **RRF** (không dùng alpha cộng điểm) | RRF là lựa chọn đúng về mặt kỹ thuật (BM25 score và cosine không cùng thang đo — hợp với `design_doc §3.1`). Không nên quay lại alpha-weighted sum. Có thể "giả lập" ý tưởng alpha bằng **weighted RRF** (mục 8c). |
| 6. Top 20 → rerank (`thanhtantran/Vietnamese_Reranker`) → Top 5 | `rerank.py` dùng `AITeamVN/Vietnamese_Reranker`, top_k cấu hình được | Tên model khác nhưng cùng họ; giữ `AITeamVN` (đồng bộ với embedding, đã note trong `rag-system-design.md §5`) là hợp lý hơn, không cần đổi. |
| 7. Prompt cuối cùng gọi `Qwen3.5-0.8B` | Repo hiện dùng `Qwen/Qwen3-8B` | **Phải đổi**, xem mục 2. |

### Đối chiếu `guideline.txt`

- ✅ Model ≤10B, open-weight, không dùng ChatGPT/Claude/Gemini — repo tuân thủ (Qwen).
- ✅ Rate limit 1 req/5s — `case_api_client.py` đã implement đúng qua `_RateLimiter`.
- ⚠️ **Nhãn 4 lớp** — repo đã hỗ trợ đúng 4 nhãn (`VALID_PREDICTIONS`), không cần sửa.
- 🔴 **Private Test format tối giản** (chỉ `case_id` + `case_query`) — đây là thay đổi quan trọng nhất cần code phải "sống được" mà không có `n_segments`, `related_law_provisions`, v.v. Cụ thể:
  - `submission.py` đã defensive-code đúng (`c.get("n_segments") or c.get("n_i")` → `None` là ổn).
  - Nhưng `_case_api_budget()` cần một **default budget thông minh hơn số cố định 8**, xem mục 5.
  - `test_all_backend.py` hiện dựa vào `gold["verdict_label"]` và `related_law_provisions` — các trường này **chỉ có ở Public Test**, không có ở Private Test. Việc này không sao vì đó là script test nội bộ, nhưng cần ghi rõ comment để không ai nhầm là format chính thức.
- ⚠️ Giới hạn nộp bài 3 lần/ngày lên leaderboard — không liên quan code, nhưng nên thêm 1 dòng log cảnh báo trong `submission.py` nhắc nhở trước khi ai đó chạy `main()` nhiều lần trong ngày (tránh lãng phí lượt nộp).

---

## 2. Chuyển sang Qwen3.5-0.8B — tránh AttributeError / OOM

### Rủi ro cụ thể khi đổi model trong code hiện tại

1. **`device_map=device if device.startswith("cuda") else None`** — dùng `device_map="cuda"` như một string là cách làm dễ vỡ giữa các version `transformers`/`accelerate` (đôi khi ném `ValueError`/`AttributeError` từ `accelerate.dispatch_model` khi wrap thêm hook). Với model 0.8B, không cần `device_map` (vốn sinh ra để *shard* model lớn qua nhiều GPU) — nên bỏ hẳn, load thường rồi `.to(device)`.
2. **`torch_dtype=...`** đang bị deprecate dần sang `dtype=...` ở các bản `transformers` mới — nếu bản cài trong môi trường target đã bỏ `torch_dtype`, sẽ gây `TypeError`/warning leo thang. Nên thử `dtype` trước, fallback `torch_dtype`.
3. **`apply_chat_template(..., add_generation_prompt=True)`** không truyền `enable_thinking=False`. Họ Qwen3 (và nhiều khả năng Qwen3.5) có chế độ reasoning bật mặc định → sinh `<think>...</think>` dài, dễ vượt `max_new_tokens` trước khi ra JSON → `_extract_json` fail → rơi vào `_safe_default` (không phải bug crash, nhưng **âm thầm hạ chất lượng toàn bộ hệ thống về toàn B_WIN**, đúng như log trong `test_all_backend.py` từng cảnh báo). Cần tắt tường minh, có fallback nếu tokenizer không hỗ trợ kwarg này.
4. Model nhỏ đôi khi không có `chat_template` sẵn (nếu tải nhầm bản base thay vì `-Instruct`) → `apply_chat_template` ném lỗi rõ ràng (không phải AttributeError âm thầm) nhưng cần try/except để không crash cả batch.
5. **Không cần quantization** (bitsandbytes) cho model 0.8B — thêm bnb chỉ tăng bề mặt lỗi (yêu cầu CUDA compute capability, có thể fail build trên môi trường lạ) mà lợi ích gần như bằng 0 với model đã rất nhỏ.
6. **`attn_implementation`**: không set → mặc định có thể cố gắng dùng `flash_attention_2` nếu có sẵn nhưng thiếu gói `flash-attn` sẽ crash khi import. Nên ép `sdpa` (built-in PyTorch, không cần cài thêm) để giảm rủi ro.
7. **Ngân sách bộ nhớ thực tế trên T4 15GB / CPU 12GB**:
   - Qwen3.5-0.8B fp16 ≈ 1.6GB trọng số + KV-cache nhỏ (context ngắn) → an toàn.
   - `AITeamVN/Vietnamese_Embedding` (~560M) ≈ 1.1–2.2GB tùy fp32/fp16.
   - `AITeamVN/Vietnamese_Reranker` (cùng cỡ) ≈ 1–2GB.
   - Tổng cộng < 6GB trên GPU, còn dư nhiều margin cho activation/batch — **OOM sẽ biến mất** sau khi đổi model sinh, miễn là 3 model không được load bằng fp32 mặc định trên GPU (ép fp16 rõ ràng).
   - Trên CPU 12GB: nên ép `torch_dtype=float32` (fp16 trên CPU thường chậm/không được hỗ trợ tốt bởi mọi op) và giảm `batch_size` embed xuống 8–16 nếu chạy CPU.

### Code đề xuất (`backend/config.py`)

```python
GENERATION_MODEL_NAME = _env("GENERATION_MODEL_NAME", "Qwen/Qwen3.5-0.8B-Instruct")  # xác nhận đúng repo id trên HF trước khi chạy thật
HYDE_MODEL_NAME = _env("HYDE_MODEL_NAME", GENERATION_MODEL_NAME)  # không còn dùng sau khi bỏ HyDE (mục 3), có thể xoá dòng này
GENERATION_ENABLE_THINKING = _env("GENERATION_ENABLE_THINKING", "false").lower() == "true"
GENERATION_ATTN_IMPL = _env("GENERATION_ATTN_IMPL", "sdpa")
GENERATION_MAX_NEW_TOKENS_DEFAULT = _env_int("GENERATION_MAX_NEW_TOKENS_DEFAULT", 500)
```

### Code đề xuất (`backend/models.py`, thay `_get_generation_model` và `generate_text`)

```python
@lru_cache(maxsize=1)
def _get_generation_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        config.GENERATION_MODEL_NAME, trust_remote_code=True
    )
    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"{config.GENERATION_MODEL_NAME} không có chat_template — "
            "kiểm tra đã trỏ đúng bản '-Instruct'/'-Chat', không phải bản base."
        )

    load_kwargs = dict(
        trust_remote_code=True,
        attn_implementation=config.GENERATION_ATTN_IMPL,
    )
    # torch_dtype -> dtype rename giữa các version transformers: thử cả hai.
    dtype_val = torch.float16 if device.startswith("cuda") else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.GENERATION_MODEL_NAME, dtype=dtype_val, **load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            config.GENERATION_MODEL_NAME, torch_dtype=dtype_val, **load_kwargs
        )

    model = model.to(device)  # không dùng device_map cho model <1B tham số
    model.eval()
    return tokenizer, model, device


def generate_text(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    import torch

    tokenizer, model, device = _get_generation_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    template_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, enable_thinking=config.GENERATION_ENABLE_THINKING, **template_kwargs
        ).to(device)
    except TypeError:
        # tokenizer/chat template không hỗ trợ enable_thinking (model khác họ Qwen3) — fallback an toàn.
        input_ids = tokenizer.apply_chat_template(messages, **template_kwargs).to(device)

    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Nếu model vẫn lọt <think>...</think> ra ngoài (fallback path), cắt bỏ trước khi trả về.
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text
```

> **Lưu ý xác minh**: tên chính xác trên Hugging Face (`Qwen/Qwen3.5-0.8B` hay `Qwen/Qwen3.5-0.8B-Instruct`) cần được kiểm tra thủ công trước khi chạy — README hiện ghi `Qwen/Qwen3.5-0.8B` không có hậu tố Instruct, nếu đó là bản base (không chat template) thì đoạn code check `chat_template is None` ở trên sẽ báo lỗi rõ ràng ngay từ đầu thay vì để `AttributeError` mơ hồ rơi xuống tận `generate_text`.

---

## 3. Bỏ HyDE — thay bằng NER + Query Decomposition có kiểm soát

### Vì sao bỏ HyDE là hợp lý
`rag-system-design.md §2.3` tự ghi rõ rủi ro: LLM có thể "bịa" nội dung nghe giống văn phong luật nhưng sai, rồi chính đoạn bịa đó lại được dùng làm vector query. Với model đã giảm xuống 0.8B tham số, rủi ro hallucination-trong-query càng tăng (model nhỏ hơn dễ sinh "văn phong luật giả" kém chính xác hơn). Đồng thời cả hai paper đính kèm đều **không dùng HyDE**:
- ViDRILL (VLSP 2025 DRiLL) đạt top-5 chỉ bằng BM25 + dense (E5-Instruct/GTE/BGE-M3) + cross-encoder rerank + fallback filtering — không có bước sinh câu trả lời giả định nào.
- Judge-R1 thay HyDE bằng một **agent phân rã câu hỏi thành nhiều sub-query theo khía cạnh pháp lý** (constitutive elements, sentencing factors) rồi tổng hợp bằng weighted RRF — cách này cải thiện Recall rất mạnh (P@5 tăng từ 0.15 (BM25) lên 0.64 (agentic), Table 1 trong `2605.02011v1.pdf`).

### Đề xuất thay thế: NER-grounded Query Decomposition (không sinh văn bản luật giả định)

Ý tưởng: dùng **NER thật** (`NlpHUST/ner-vietnamese-electra-base`, đúng như `pipeline.txt` bước 2 đã định hướng) để tách case_query thành các thực thể/khía cạnh, sau đó **lắp ráp** (không generate tự do) thành các sub-query tường minh:

```python
# backend/retrieval/ner.py
from __future__ import annotations
from functools import lru_cache

NER_MODEL_NAME = "NlpHUST/ner-vietnamese-electra-base"

@lru_cache(maxsize=1)
def _get_ner_pipeline():
    from transformers import pipeline
    return pipeline("token-classification", model=NER_MODEL_NAME,
                     aggregation_strategy="simple")

def extract_entities(text: str) -> list[dict]:
    """Trả về [{"text":..., "entity_group":..., "score":...}, ...].
    Fail-safe: lỗi tải model không được làm sập cả pipeline truy hồi."""
    try:
        return _get_ner_pipeline()(text)
    except Exception:
        return []

def mask_person_org_entities(text: str, entities: list[dict] | None = None) -> str:
    """Che tên riêng/tổ chức — dùng khi build query cho LAW retrieval,
    vì tên nguyên đơn/bị đơn là nhiễu đối với tìm kiếm điều luật."""
    entities = entities if entities is not None else extract_entities(text)
    spans = sorted(
        [e for e in entities if e.get("entity_group") in ("PERSON", "ORGANIZATION")],
        key=lambda e: e["start"], reverse=True,
    )
    for e in spans:
        text = text[: e["start"]] + "[BÊN LIÊN QUAN]" + text[e["end"]:]
    return text
```

```python
# backend/retrieval/querry_transform.py — thay generate_hyde() bằng decompose_query()
_DECOMPOSE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Cho tình huống dưới đây, hãy liệt kê 3-4 khía cạnh pháp lý "
    "riêng biệt cần tra cứu (ví dụ: quan hệ pháp luật tranh chấp, điều kiện có hiệu lực "
    "của hợp đồng/giao dịch, thời hiệu khởi kiện, nghĩa vụ chứng minh). "
    "Mỗi khía cạnh viết thành MỘT câu hỏi ngắn, không đánh số, không giải thích thêm, "
    "KHÔNG được tự bịa nội dung điều luật cụ thể — chỉ nêu khía cạnh cần tra."
)

def decompose_query(query: str, masked_query: str | None = None,
                     n_subqueries: int = 4, max_new_tokens: int = 200) -> list[str]:
    """Thay thế generate_hyde(): phân rã câu hỏi thành các khía cạnh pháp lý cần tra,
    KHÔNG sinh văn bản điều luật giả định (tránh rủi ro hallucination của HyDE).
    """
    try:
        raw = generate_text(
            system_prompt=_DECOMPOSE_SYSTEM_PROMPT,
            user_prompt=masked_query or query,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
        )
    except Exception:
        return []
    lines = [l.strip("-•\t ") for l in raw.splitlines() if l.strip()]
    return [l for l in lines if len(l) > 5][:n_subqueries]
```

```python
# backend/retrieval/hybrid_search.py — bỏ use_hyde, dùng decomposition + weighted RRF
from backend.retrieval.ner import extract_entities, mask_person_org_entities
from backend.retrieval.querry_transform import decompose_query, rewrite_query

def hybrid_search(query: str, law_id=None, require_active=True,
                   use_query_rewriting=True, use_decomposition=True, top_k=30):
    bm25 = bm25_index.get_bm25_index()

    entities = extract_entities(query)
    law_query = mask_person_org_entities(query, entities)  # che tên riêng cho tìm luật

    channels: list[tuple[list, float]] = []  # (result_list, weight) cho weighted RRF

    base_queries = rewrite_query(law_query) if use_query_rewriting else [law_query]
    for q in base_queries:
        r1 = bm25.query(q, top_k=config.BM25_TOP_K, law_id=law_id, require_active=require_active)
        r2 = vector_store.query(q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active)
        channels.append((r1, 1.0))
        channels.append((r2, 1.0))

    if use_decomposition:
        for sub_q in decompose_query(query, masked_query=law_query):
            r = vector_store.query(sub_q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active)
            channels.append((r, 2.0))  # w_agent=2.0 theo Judge-R1 (đặt câu tách khía cạnh có trọng số cao hơn)

    fused = _rrf_fuse_weighted(channels)
    return fused[:top_k]
```

`_rrf_fuse_weighted` chỉ là bản mở rộng của `_rrf_fuse` hiện có, nhân thêm `weight` vào từng hạng:
`scores[cid] += weight / (k + rank)`.

Ưu điểm so với HyDE:
- Không có bước sinh "văn bản luật giả định" → loại bỏ hoàn toàn rủi ro hallucination-trong-truy-vấn mà `design_doc` từng cảnh báo.
- Tận dụng đúng bước NER mà `pipeline.txt` đã định hướng nhưng repo chưa làm.
- Có cơ sở thực nghiệm rõ ràng từ Judge-R1 (P@5 0.15→0.64 khi thay truy vấn tĩnh bằng truy vấn phân rã theo khía cạnh).
- Rẻ hơn HyDE một chút (không cần sinh đoạn văn dài mô phỏng văn phong luật, chỉ cần liệt kê khía cạnh ngắn).

---

## 4. `test/test_all_backend.py` — precision/recall đã có chưa? Có nên dùng ragas?

### Hiện trạng
- **Đã có** `compute_law_f1()` tính Precision/Recall/F1 (khớp `article_num`/`aid`), nhưng:
  - 🔴 Đây là **macro F1** (trung bình `f1` từng case), trong khi `evaluation.md §2.6` yêu cầu **micro F1** (gộp `|P∩G|`, `|P|`, `|G|` toàn tập trước, chia sau). Hai chỉ số này không tương đương và có thể lệch nhiều nếu số lượng gold provisions không đều giữa các case.
  - Không có phép tính nào cho `PenalizedCaseRecall` (không có gold `case_evidence` trong public test — script đã tự ghi chú điều này, đúng).

### Patch đề xuất (bổ sung micro F1 đúng công thức, giữ macro để tham khảo)

```python
# test/test_all_backend.py — trong vòng lặp, gom thêm 2 set toàn cục
global_pred_aids: list[int] = []
global_gold_aids: list[int] = []
...
        global_pred_aids.extend(int(p["aid"]) for p in record_dict["law_evidence"])
        global_gold_aids.extend(g["article_num"] for g in gold_provisions)
...
# --- sau vòng lặp, TRƯỚC phần summary hiện có ---
from collections import Counter
pred_ctr, gold_ctr = Counter(global_pred_aids), Counter(global_gold_aids)
tp = sum(min(pred_ctr[a], gold_ctr[a]) for a in pred_ctr)
micro_p = tp / sum(pred_ctr.values()) if pred_ctr else 0.0
micro_r = tp / sum(gold_ctr.values()) if gold_ctr else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
log.info("Micro Law F1 (đúng công thức evaluation.md §2.6): P=%.3f R=%.3f F1=%.3f",
          micro_p, micro_r, micro_f1)
log.info("(Lưu ý: đây vẫn là xấp xỉ theo article_num vì public test không cho law_id)")
```

Đồng thời thêm ước lượng `E_i` cục bộ (dùng `DEFAULT_MAX_API_CALLS_PER_CASE` làm baseline vì không có `n_i` thật — xem mục 5) để theo dõi hiệu năng API trước khi nộp:

```python
budget_guess = config.DEFAULT_MAX_API_CALLS_PER_CASE
e_i_values = [max(0.0, 1 - max(0, c - 2*budget_guess) / (3*budget_guess)) for c in (r["api_calls"] for r in results)]
log.info("Ước lượng E_i trung bình (giả định n_i≈budget_guess=%d): %.3f", budget_guess, sum(e_i_values)/len(e_i_values))
```

### Ragas có nên cài không?

Đã kiểm tra: `ragas` **có sẵn trên PyPI** (bản mới nhất 0.4.3, cài được qua `pip install ragas --break-system-packages`). Tuy nhiên:

| Yếu tố | Đánh giá |
|---|---|
| Phụ thuộc | ragas kéo theo `langchain`, `langchain-core`, và (với các metric như `Faithfulness`, `AnswerRelevancy`) cần một **LLM-as-judge** — mâu thuẫn trực tiếp với triết lý "không dùng framework RAG" đã nêu ở đầu `rag-system-design.md`. |
| Metric cần LLM judge | Không dùng được "miễn phí" — phải wrap Qwen3.5-0.8B qua interface LLM của LangChain (`BaseChatModel` custom), tốn thêm code, và một model 0.8B làm "giám khảo" đánh giá chính output của chính nó (hoặc model tương tự) cho độ tin cậy thấp. |
| Metric không cần LLM (`NonLLMContextPrecisionWithReference`, `NonLLMContextRecall`) | Có thể dùng được (chỉ so khớp string/embedding giữa context truy hồi và context tham chiếu), nhưng về bản chất **chính là công thức Precision/Recall đã tự viết trong `compute_law_f1`** — không mang lại thông tin mới, chỉ thêm dependency nặng. |
| Độ khớp với công thức chấm điểm thật của ALQAC | Công thức tự viết bám sát 100% `evaluation.md`; ragas là thư viện tổng quát, không biết về công thức `E_i` (API efficiency penalty) đặc thù của cuộc thi này — vẫn phải tự code phần đó dù có ragas hay không. |

**Kết luận: không nên thêm ragas vào pipeline chính.** Thay vào đó, mở rộng harness tự viết hiện có (đã đúng hướng) để: (a) sửa macro→micro F1 cho khớp công thức thật, (b) thêm theo dõi `E_i` ước lượng. Nếu muốn có thêm một góc nhìn "đối chiếu độc lập" trong giai đoạn R&D (không phải để nộp bài), có thể cài `ragas` trong một notebook tách biệt, chỉ dùng 2 metric non-LLM nói trên, không đưa vào `test/` chính thức.

---

## 5. Context window ~10k token — tách pipeline case-retrieve / law-retrieve / generation

### Chẩn đoán nguồn gốc thật của 10k token
Không phải do HyDE/query rewriting (các lệnh gọi này input chỉ là `case_query`, vài chục–trăm token). Nguồn thật: `prompt_builder.build_prediction_prompt()` gộp **toàn bộ** law chunks (verbatim, không nén — đúng quy tắc §4.2) + **toàn bộ** case-evidence hits (đã nén nhưng vẫn có thể nhiều đoạn) vào **một lần gọi LLM duy nhất** ở cuối. Với `FINAL_LAW_TOP_K=5` (mỗi Điều/Khoản ~200–1000 token) + nhiều case-evidence chunk, tổng dễ chạm 10k.

### Giải pháp: tách 2 pipeline truy hồi (đã tách sẵn) + THÊM 1 bước tổng hợp trung gian trước khi vào prompt cuối

Truy hồi (`collect_case_evidence` và `collect_law_evidence`) *đã* độc lập với nhau về mặt code — không cần tách thêm. Cái cần tách là **generation**: hiện tại 1 prompt làm 2 việc (tóm tắt tình tiết + suy luận pháp lý). Đề xuất chia thành 2 lệnh gọi LLM nhỏ, tổng token thấp hơn nhiều so với 1 lệnh gọi lớn:

```
Case evidence hits (nhiều, dài)  ──►  [LLM #1: Case-Fact Digest]  ──► digest ngắn (~150-250 token)
                                                                          │
case_query + digest + law_chunks (verbatim) ────────────────────►  [LLM #2: Verdict Generator]  ──► JSON kết quả
```

```python
# backend/generation/case_digest.py  (mới)
from backend.models import generate_text

_DIGEST_SYSTEM_PROMPT = (
    "Bạn là trợ lý tóm tắt hồ sơ vụ án. Từ các đoạn bằng chứng vụ án dưới đây, hãy viết "
    "một đoạn TÓM TẮT NGẮN (tối đa 150 từ) nêu: các bên, yêu cầu của nguyên đơn, "
    "lập luận/bằng chứng chính của mỗi bên. KHÔNG suy đoán, KHÔNG thêm thông tin ngoài "
    "các đoạn được cung cấp. Nếu bằng chứng trống, trả lời đúng câu: "
    "'(Không có bằng chứng vụ án được truy hồi.)'"
)

def build_case_digest(case_query: str, evidence_texts: list[str]) -> str:
    if not evidence_texts:
        return "(Không có bằng chứng vụ án được truy hồi.)"
    joined = "\n\n".join(f"- {t.strip()}" for t in evidence_texts)
    user_prompt = f"TÌNH HUỐNG:\n{case_query.strip()}\n\nBẰNG CHỨNG:\n{joined}"
    try:
        return generate_text(
            system_prompt=_DIGEST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=220,
            temperature=0.2,
        ).strip()
    except Exception:
        # Không để lỗi ở bước digest làm sập cả pipeline — dùng bản nén cũ làm fallback.
        return "\n".join(evidence_texts)[:800]
```

`prompt_builder.py` sửa lại để nhận `case_digest: str` thay vì `case_evidence_hits` đầy đủ:

```python
def build_prediction_prompt(case_query, law_chunks, case_digest: str) -> tuple[str, str]:
    law_section = _format_law_section(law_chunks)
    user_prompt = f"""TÌNH HUỐNG VỤ ÁN (case_query):
{case_query.strip()}

TÓM TẮT BẰNG CHỨNG VỤ ÁN (đã được tổng hợp):
{case_digest.strip()}

CÁC ĐIỀU LUẬT LIÊN QUAN ĐÃ TRUY HỒI (nguyên văn — chỉ được trích dẫn trong danh sách này):
{law_section}

Hãy đưa ra dự đoán kết quả vụ án theo đúng schema JSON đã quy định."""
    return SYSTEM_PROMPT, user_prompt
```

Điểm quan trọng: **case_evidence nộp lên submission (`case_evidence: [chunk_id, ...]`) không nhất thiết phải bằng số đoạn đưa vào LLM #1.** Có thể thu thập nhiều hơn (để tối đa hoá `Recall_i^case`, vẫn tính vào `E_i` như cũ) nhưng chỉ đưa top-N theo `score` (ví dụ top 5) vào `build_case_digest` để kiểm soát token — tách rõ "evidence để chấm điểm" và "evidence để sinh câu trả lời":

```python
# backend/pipeline.py
TOP_N_EVIDENCE_FOR_DIGEST = 5

def process_case(case: CaseQuery) -> SubmissionRecord:
    case_evidence_hits = collect_case_evidence(case)                      # dùng đầy đủ cho scoring
    top_hits_for_digest = sorted(case_evidence_hits, key=lambda h: h.score, reverse=True)[:TOP_N_EVIDENCE_FOR_DIGEST]
    digest = build_case_digest(case.case_query, [h.text for h in top_hits_for_digest])

    law_chunks = collect_law_evidence(case.case_query)
    outcome = predict_outcome(case.case_query, law_chunks, digest)
    ...
```

Ước lượng token sau khi tách: LLM #1 input ~top-5 evidence (đã nén) ~1–2k token, output ~250 token. LLM #2 input = case_query (~100) + digest (~250) + 5 law chunks (~1000–3000, tuỳ `FINAL_LAW_TOP_K`) → tổng **~2–4k token cho lệnh gọi lớn nhất**, so với ~10k trước đó — giảm ~60–70%, đồng thời **phù hợp hơn với context hiệu quả của một model 0.8B** (model càng nhỏ càng "loãng" chất lượng khi context dài).

Khuyến nghị thêm: hạ `FINAL_LAW_TOP_K` từ 5 xuống 3–4 nếu vẫn thấy token cao, vì marginal recall gain của provision thứ 5 thường thấp trong khi chi phí token/nhiễu tăng (tương tự nhận xét của ViDRILL: "thêm quá nhiều dense retriever tạo nhiễu, giảm precision").

---

## 6. Chunking: giữ rule-based nhưng chuyển sang **dual-granularity** (theo ViDRILL), và **fix lỗi wiring parent-lookup**

### Không nên đổi sang Semantic Chunking thuần
`rag-system-design.md §2.2` đã tự đặt nguyên tắc: **không cắt theo token/độ tương đồng ngữ nghĩa** vì có thể tách rời một điều khoản khỏi cụm từ loại trừ đứng trước nó ("trừ trường hợp..."). Semantic chunking (dựa vào embedding similarity để tìm điểm cắt) có đúng loại rủi ro này — nó tối ưu cho *sự mạch lạc ngữ nghĩa*, không phải cho *ranh giới pháp lý chính xác*. Với domain mà một từ nối có thể đảo nghĩa cả điều khoản, rule-based structural chunking (Chương>Mục>Điều>Khoản>Điểm) vẫn là lựa chọn an toàn hơn và nên **giữ nguyên**.

### Điều nên đổi: áp dụng **dual-level chunking như ViDRILL** một cách triệt để hơn

Repo hiện tại *đã có sẵn* 2 cấp (`parent`=Điều, `child`=Khoản/Điểm) — về ý tưởng đã gần giống ViDRILL (long chunk cho rerank, short chunk cho retrieval). Nhưng có **2 khoảng trống**:

**(a) `build_parent_lookup()` không được dùng ở đâu cả** — reranker (`rerank.py`) hiện chấm điểm `(query, child.text)`, tức là context ngắn (chỉ 1 Khoản), trong khi ViDRILL cho thấy rerank trên đoạn dài (cả Điều, ~2000 ký tự) cho kết quả tốt hơn nhờ cross-encoder có nhiều ngữ cảnh hơn để đánh giá độ liên quan. **Sửa**: trước khi rerank, thay `candidate.text` bằng text của parent chunk tương ứng (giữ nguyên `chunk_id`/`aid` của child để trích dẫn chính xác):

```python
# backend/pipeline.py
from backend.ingestion.chunker import build_parent_lookup  # cần expose parent map ra ngoài (persist cùng BM25 index hoặc load riêng)

def collect_law_evidence(query_text: str):
    candidates = hybrid_search(query_text, top_k=config.RERANK_TOP_K)
    enriched = []
    for c in candidates:
        parent = PARENT_LOOKUP.get(f"{c.law_id}_a{c.aid}")  # map parent_id -> LawChunk, nạp 1 lần khi khởi động
        rerank_text = parent.text if parent else c.text
        enriched.append(c.model_copy(update={"text": rerank_text}))
    reranked = rerank(query_text, enriched, top_k=config.FINAL_LAW_TOP_K)
    # Sau khi rerank xong, trả child.text gốc lại cho phần trích dẫn trong prompt (ngắn gọn, đúng Khoản)
    # hoặc giữ parent.text nếu muốn LLM có full ngữ cảnh Điều — tuỳ đánh đổi token vs độ chính xác trích dẫn.
    return reranked
```

> Đánh đổi cần cân nhắc: nếu giữ parent-text luôn cho cả bước generation cuối, token lại tăng trở lại (mâu thuẫn mục 5). Khuyến nghị: **chỉ dùng parent-text cho bước rerank** (không đưa vào prompt cuối), sau rerank thì swap lại về child-text (ngắn, chính xác theo Khoản) trước khi đưa vào `prompt_builder`. Đây là lý do nên thêm 1 field tạm `rerank_text` khác với `text` gốc trong luồng nội bộ, tránh nhầm lẫn.

**(b) Oversized child chunk hiện bị "giữ nguyên không cắt"** (`chunker.chunk_articles` docstring: "kept intact rather than force-split"). ViDRILL xử lý tốt hơn: khi một Khoản/Điểm vượt quá ngưỡng ký tự (450 ký tự trong paper), họ tiếp tục cắt theo **ranh giới câu/cụm coherent** (không cắt cứng theo token). Đề xuất bổ sung 1 fallback splitter tương tự, chỉ kích hoạt khi 1 Khoản/Điểm vượt ngưỡng (ví dụ `CHILD_CHUNK_MAX_TOKENS`), cắt theo dấu câu (`.`, `;`) chứ không theo token — giữ đúng tinh thần "không cắt cứng":

```python
CHILD_MAX_CHARS = 900  # tương đương ~450 từ tiếng Việt, tham khảo ViDRILL

def _soft_split_oversized(text: str, max_chars: int = CHILD_MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.;])\s+", text)
    out, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) > max_chars and buf:
            out.append(buf.strip())
            buf = s
        else:
            buf += (" " if buf else "") + s
    if buf:
        out.append(buf.strip())
    return out or [text]
```

Kết luận mục 6: **giữ kiến trúc Hierarchical rule-based hiện tại (không chuyển sang Semantic Chunking)**, nhưng (i) sửa lỗi wiring để parent chunk thực sự được dùng ở bước rerank như thiết kế ban đầu đã dự tính, và (ii) thêm soft-split cho Khoản/Điểm quá dài, theo đúng thông số đã được kiểm chứng thực nghiệm trong ViDRILL (cùng domain pháp lý tiếng Việt, cùng kiểu tổ chức shared-task với ALQAC).

---

## 7. Đánh giá các kỹ thuật bổ sung

| Kỹ thuật | Khả thi? | Nhận xét & đề xuất cụ thể |
|---|---|---|
| **Late chunking (BambiBert)** | ⚠️ Khả thi có điều kiện | Late chunking (embed cả văn bản dài trước, pool token-embedding theo ranh giới chunk sau) cần backbone hỗ trợ context dài + truy cập token-level embedding (không phải API `.encode()` thông thường của `sentence-transformers`). `AITeamVN/Vietnamese_Embedding` (nền BGE-M3) *có* hỗ trợ context tới 8192 token nên về lý thuyết dùng được. Tuy nhiên tên "BambiBert" không phải checkpoint phổ biến/đã xác minh — cần đội ngũ tự kiểm tra model card trước khi đầu tư. Chi phí kỹ thuật (viết lại tầng embed để lấy token-embedding + pooling tùy biến theo từng ranh giới Khoản/Điểm) khá cao so với lợi ích chưa được đo trên domain pháp lý VN. **Đề xuất: để giai đoạn R&D sau, không đưa vào core pipeline ngay**, ưu tiên các mục có ROI rõ ràng hơn bên dưới trước. |
| **Neurosymbolic** | ✅ Khả thi ở mức nhẹ | Không cần cả một engine suy luận symbolic đầy đủ. Có thể thêm **lớp hậu-kiểm rule-based** sau khi parse JSON (tương tự `allowed_citation_keys` đã có để chặn hallucinated citation): nếu số `law_citations` sau lọc = 0 hoặc `case_evidence` rỗng, ép `confidence` xuống ngưỡng thấp và cân nhắc downgrade nhãn về phía `B_WIN`/`PARTIAL_B_WIN` (thận trọng hơn) thay vì tin hoàn toàn vào self-report "hạ confidence" của LLM (prompt rule #3 hiện chỉ yêu cầu LLM tự giác, không có gì ép buộc). Đây là mở rộng rẻ, dễ implement trong `generate.py`, tăng độ tin cậy. |
| **Retrieval Evaluator → core cho vòng lặp** | ✅ Rất khuyến nghị, chi phí thấp | Không cần thêm LLM-judge riêng — **tái sử dụng chính điểm cross-encoder rerank đã có** làm evaluator (giống thiết kế Step 3 của ViDRILL: ngưỡng 0.75, fallback top-40). Nếu điểm rerank cao nhất < ngưỡng, trigger thêm 1 vòng `decompose_query` + re-retrieve trước khi vào generation. Đây gần như miễn phí (không gọi thêm LLM lớn) và bám sát đúng ý "Retrieval Evaluator -> Core for looping" mà không phát sinh model mới hay chi phí GPU thêm. |
| **Query Advanced Preprocessing** | ✅ Đã đề xuất ở mục 3 | Tách 2 biến thể truy vấn: (i) bản đầy đủ (giữ tên riêng, tình tiết) cho Case Content API — vì API cần "keywords describing evidence", chi tiết càng cụ thể càng tốt; (ii) bản đã che tên riêng (`mask_person_org_entities`) cho law retrieval — vì tên nguyên/bị đơn là nhiễu thuần tuý đối với tìm điều luật áp dụng. |
| **NER** | ✅ Đã đề xuất ở mục 3 | `NlpHUST/ner-vietnamese-electra-base` dùng cho (i) masking tên riêng và (ii) tiềm năng gắn nhãn loại quan hệ pháp luật/tài sản tranh chấp để làm giàu breadcrumb hoặc query — đúng như `pipeline.txt` bước 2 định hướng nhưng repo chưa làm. |
| **Weighted RRF theo Judge-R1** | ✅ Bổ sung nhỏ, có cơ sở thực nghiệm | Judge-R1 dùng `w_agent=2.0 / w_std=1.0` khi fuse route truy vấn gốc và route truy vấn đã lập kế hoạch (agentic), vì route agentic cho recall cao hơn hẳn (Table 1: P@5 0.64 vs 0.15 BM25 thuần). Đã tích hợp vào code đề xuất ở mục 3 (`_rrf_fuse_weighted`). |

---

## 8. Thứ tự triển khai đề xuất (ưu tiên theo rủi ro/chi phí)

1. **Đổi model sinh sang Qwen3.5-0.8B** với code loading an toàn (mục 2) — bắt buộc để tránh OOM, làm trước tiên.
2. **Bỏ HyDE, thêm NER + query decomposition + weighted RRF** (mục 3) — giảm rủi ro hallucination-trong-truy-vấn, tăng recall có cơ sở thực nghiệm.
3. **Tách generation thành 2 bước (digest → verdict)** để giải quyết context 10k token (mục 5) — cải thiện tốc độ + chất lượng cho model nhỏ.
4. **Fix wiring `build_parent_lookup` vào rerank + soft-split oversized Khoản** (mục 6).
5. **Thêm retrieval-evaluator loop tái dùng điểm rerank** (mục 7) — chi phí thấp, lợi ích cao.
6. **Fix micro F1 trong `test_all_backend.py` + thêm ước lượng `E_i`** (mục 4) — để số liệu local phản ánh đúng công thức chấm điểm thật, tránh ảo tưởng về chất lượng hệ thống trước khi nộp bài thật.
7. Rule-based neurosymbolic post-check (confidence/label downgrade) — tùy chọn, làm sau khi các bước trên ổn định.
8. Late chunking (BambiBert) — để nghiên cứu sau, không phải ưu tiên cho vòng nộp bài sắp tới.

Không áp dụng: đổi sang Semantic Chunking thuần (mục 6), thêm `ragas` vào pipeline chấm điểm chính thức (mục 4) — cả hai đi ngược lại các nguyên tắc thiết kế đã tự đặt ra trong `rag-system-design.md` mà không mang lại lợi ích rõ ràng tương xứng với rủi ro/chi phí.
