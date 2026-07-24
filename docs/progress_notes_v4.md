# LegalRAG — Kế hoạch thực hiện

**Mục đích:** gộp toàn bộ phát hiện + đề xuất từ 3 lần phân tích log (`test_all_backend.log` gốc, `submission_pri.log`, `test_all_backend (2).log`) và phần thảo luận di trú sang Llama-3.1-8B-Instruct/NVIDIA NIM, thành **một danh sách việc cần làm duy nhất, có trạng thái, có patch code, có tiêu chí chấp nhận** — để một coding agent hoặc người kế nhiệm đọc là bắt tay làm được ngay, không cần lục lại toàn bộ lịch sử hội thoại.

**Quan hệ với các tài liệu khác:**
- `docs/system_adjustments_v4.md` — vẫn là nguồn tham chiếu chẩn đoán gốc cho 4 mục 3.1–3.4.
- `README.md`, `requirements.txt`, `docs/test_design.md` — **cần cập nhật lại theo Nhóm D bên dưới**, việc này đã bắt đầu nhưng chưa hoàn tất.

---

## 0. Bảng trạng thái tổng quan (tính đến lần chạy gần nhất)

| # | Việc | Trạng thái | Bằng chứng | Nhóm |
|---|---|---|---|---|
| 3.1 | Cache Pinecone Index client | ✅ **Đã xong** | `test_all_backend (2).log`: "Listing indexes" chỉ xuất hiện 1 lần/run | — |
| 3.2 | Sửa đếm API call khi retry | ✅ **Đã xong** | Không còn case nào `api_calls > 8`; code có comment "§3.2 fix applied here" | — |
| 3.3 | Sửa lệch namespace `aid` (gold vs pred) | ⚠️ **Dở dang** — code đúng hướng nhưng không hiệu quả | `aid_map` chỉ map được 245 entries, Micro Law F1 vẫn = 0.000 | A |
| 3.4 | Sửa sụp nhãn dự đoán (bước 1: instrumentation) | ✅ **Đã xong** | Log có `is_fallback`/`fallback_reason`, fallback rate=4.0% | — |
| 3.4 | Sửa sụp nhãn dự đoán (bước 2: nguyên nhân gốc) | ❌ **Phát hiện bug mới, chưa sửa** | 33/50 case bị ép từ A_WIN → PARTIAL_A_WIN do ngưỡng `ratio>0.99` quá khắt khe | A |
| — | Xử lý mã lỗi HTTP lạ (404...) trong `case_api_client.retrieve()` | ❌ **Chưa sửa** | `submission_pri.log`: 404 làm sập toàn bộ 1 case thay vì chỉ 1 query | A |
| — | JSON parse fail do thiếu dấu phẩy trong mảng `law_citations` | ❌ **Chưa sửa** | 3/60 + 2/50 case lỗi đúng dạng "Expecting ',' delimiter" ở dòng chứa citation | A |
| — | Di trú sang Llama-3.1-8B-Instruct qua NVIDIA NIM + structured output | ❌ **Chưa bắt đầu** (đã thiết kế, chưa code) | — | B |
| — | Cập nhật README.md / requirements.txt / test_design.md theo mã nguồn thực tế | ⚠️ **Dở dang** | README.md ghi đè bị lỗi "File already exists"; requirements.txt và test_design.md chưa viết lại | D |

---

## Nhóm A — Fix nhanh, KHÔNG phụ thuộc việc đổi model (làm trước tiên, ROI cao nhất)

Các mục này độc lập với Nhóm B — nên làm ngay cả khi quyết định không di trú sang NIM, vì đây là bug logic thuần, không liên quan năng lực model.

### A1. Hạ ngưỡng suy nhãn từ ratio — ưu tiên #1 tuyệt đối

**File:** `backend/generation/generate.py`, hàm `_label_from_ratio()`

**Hiện trạng:**
```python
def _label_from_ratio(ratio: float) -> Prediction:
    if ratio > 0.99:
        return "A_WIN"
    if ratio > 0.5:
        return "PARTIAL_A_WIN"
    if ratio > 0.0:
        return "PARTIAL_B_WIN"
    return "B_WIN"
```

**Vấn đề:** model chỉ bao giờ xuất `ratio ∈ {0.95, 0.99}` khi tự tin cao, gần như không bao giờ vượt `0.99`. Log ghi nhận **33/50 case** (66%) rơi vào nhánh:
```
categorical/ratio label mismatch: model picked 'A_WIN' but accepted_ratio_estimate=0.95/0.99 implies 'PARTIAL_A_WIN' -> using the ratio-derived label
```
→ cơ chế được thêm vào để "chống sụp nhãn" (system_adjustments_v4.md §3.4) đang **tự phá chính nhãn A_WIN** mà nó cố cứu.

**Patch:**
```python
def _label_from_ratio(ratio: float) -> Prediction:
    if ratio >= 0.9:      # hạ từ >0.99 xuống >=0.9 — model thực tế bão hoà ở 0.95/0.99
        return "A_WIN"
    if ratio > 0.5:
        return "PARTIAL_A_WIN"
    if ratio > 0.0:
        return "PARTIAL_B_WIN"
    return "B_WIN"
```
Cân nhắc thêm (không bắt buộc): log riêng phân phối `ratio` thực tế model xuất ra qua vài trăm case để chọn ngưỡng chuẩn hơn bằng dữ liệu thay vì đoán — nhưng `>=0.9` là điều chỉnh tối thiểu, an toàn dựa trên bằng chứng đã có (0.95/0.99 là 2 giá trị duy nhất quan sát được khi model chọn A_WIN).

**Acceptance criteria:**
- Chạy lại `test_all_backend.py` full 50 case, `A_WIN` trong phân phối prediction phải **> 0** (baseline hiện tại: 0).
- OutcomeAccuracy không thấp hơn 0.380 (baseline sau 3.1–3.4).

---

### A2. Bọc mã lỗi HTTP lạ (404, 5xx ngoài danh sách) thành "không có evidence" thay vì raise

**File:** `backend/case_api_client.py`, hàm `retrieve()`

**Vấn đề:** log `submission_pri.log` ghi nhận:
```
ERROR case case_6551 failed, emitting conservative fallback: 404 Client Error: Not Found for url: https://alqac-api.ngrok.pro/retrieve
```
`retrieve()` chỉ xử lý tường minh 200/429/503/403/422 — mã 404 (hoặc bất kỳ mã lạ nào khác) rơi vào `resp.raise_for_status()` → ném `requests.HTTPError` (không phải `CaseAPIError`). `pipeline.collect_case_evidence()` chỉ bắt `except CaseAPIError`, nên lỗi này **lọt thẳng lên `submission.run()`** và làm **cả case** (không chỉ 1 query) bị hy sinh về fallback rỗng.

**Patch:**
```python
def retrieve(self, query: str, case_id: str) -> CaseEvidenceHit | None:
    ...
    for attempt in range(self.max_retries):
        self._limiter.wait()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            last_exc = e
            continue

        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if not results:
                return None
            top = results[0]
            return CaseEvidenceHit(
                chunk_id=top["chunk_id"], text=top["text"], score=float(top.get("score", 0.0))
            )
        if resp.status_code == 429:
            time.sleep(config.ALQAC_MIN_REQUEST_INTERVAL_SEC)
            continue
        if resp.status_code == 503:
            time.sleep(1.0 * (attempt + 1))
            continue
        if resp.status_code == 403:
            raise CaseAPIError("403 Forbidden — missing or invalid X-API-Key.")
        if resp.status_code == 422:
            raise CaseAPIError(f"422 Malformed request: query={query!r} case_id={case_id!r}")

        # MỚI: mọi mã lỗi khác (404, 500, 502...) coi là "query này không có kết quả",
        # KHÔNG raise lên phá toàn bộ case — chỉ log cảnh báo và trả None.
        log.warning(
            "case_api_client.retrieve: unexpected status %s for case=%s query=%r — "
            "treating as no-evidence, not failing the whole case",
            resp.status_code, case_id, query,
        )
        return None

    raise CaseAPIError(f"/retrieve failed after {self.max_retries} attempts: {last_exc}")
```
(Cần thêm `import logging; log = logging.getLogger(__name__)` ở đầu file nếu chưa có.)

**Acceptance criteria:**
- Unit test mock `requests.post` trả 404 → `client.retrieve(...)` trả `None`, KHÔNG raise exception.
- Case nào từng bị mất trắng vì 404 (như `case_6551`) giờ vẫn có `case_evidence`/`law_evidence`/`prediction` bình thường từ các query khác trong cùng case.

---

### A3. Giảm tỉ lệ JSON parse fail ở mảng `law_citations`

**File:** `backend/generation/prompt_builder.py` (schema), `backend/generation/generate.py` (`_extract_json`)

**Vấn đề:** các case lỗi (`case_2035`, `case_6284`, `case_8784`, `case_2603`, `case_4584`...) đều lỗi **đúng "line 4"** với `Expecting ',' delimiter` — dòng chứa `"law_citations": [...]`, dấu hiệu model quên dấu phẩy giữa 2 object trong mảng (`{"law_id":"A","aid":1} {"law_id":"B","aid":2}`).

**Patch tạm thời (nếu chưa làm Nhóm B — vẫn dùng raw JSON parsing):** thêm bước JSON-repair nhẹ trước `json.loads` trong `_extract_json()`:
```python
def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    # JSON-repair nhẹ: chèn dấu phẩy còn thiếu giữa 2 object liền kề trong mảng,
    # lỗi điển hình của model nhỏ khi sinh mảng object lồng nhau.
    raw = re.sub(r"\}\s*\{", "}, {", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw[:200]!r}")
    return json.loads(match.group(0))
```

**Lưu ý quan trọng:** nếu Nhóm B (structured output qua `with_structured_output`) được triển khai, mục A3 này **trở nên không cần thiết** cho bước `predict_outcome` (schema ép cấu trúc ở tầng inference, không còn tự parse JSON tay) — nhưng vẫn hữu ích như một lớp phòng thủ rẻ tiền nếu Nhóm B chưa xong hoặc bị rollback.

**Acceptance criteria:**
- Test lại với input JSON cố tình thiếu dấu phẩy giữa 2 object trong mảng → `_extract_json` parse thành công thay vì raise.
- Fallback rate trong `test_all_backend.py` giảm so với baseline 4.0%.

---

### A4. Điều tra vì sao `build_aid_to_article_num_map()` chỉ map được 245 entries (chưa xong việc 3.3)

**File:** `test/test_all_backend.py`, hàm `build_aid_to_article_num_map()`

**Vấn đề:** log xác nhận map được xây (`"Built aid->article_num map from corpus: 245 entries"`), nhưng Micro Law F1 vẫn = 0.000 tuyệt đối. Hàm chỉ map thành công khi:
```python
m = re.search(r"Điều\s+(\d+)", art.title or "")
if m:
    article_num = int(m.group(1))
elif art.aid < 1000:
    article_num = art.aid
else:
    continue   # <-- KHÔNG map được, giữ nguyên aid gốc (namespace lớn) khi so khớp
```
Với phần lớn `aid` quan sát được trong submission (50854, 53238, 53359, 14428...) — đều **≥ 1000** — nếu `art.title` không chứa "Điều N" đúng định dạng, các entry này **rơi vào `continue`**, không được đưa vào map, nên khi so khớp (`_translate_predicted_aids`) sẽ fallback về giữ nguyên aid gốc → vẫn không khớp namespace nhỏ của gold.

**Bước cần làm (điều tra trước khi sửa code):**
```python
# Chạy 1 lần để xác nhận, không sửa code
import json, re
corpus = json.load(open("data/corpus_law_pub.json", encoding="utf-8"))
sample_law = corpus[0] if isinstance(corpus, list) else corpus["laws"][0]
n_with_dieu, n_total = 0, 0
for art in sample_law.get("articles", sample_law.get("content", []))[:200]:
    n_total += 1
    title = art.get("title", "")
    if re.search(r"Điều\s+\d+", title):
        n_with_dieu += 1
    else:
        print("KHÔNG match 'Điều N':", repr(title), "aid=", art.get("aid"))
print(f"{n_with_dieu}/{n_total} article có title dạng 'Điều N'")
```
Nếu tỉ lệ match thấp → cần tìm field khác trong corpus chứa số Điều thật (có thể nằm trong `content`/`body` thay vì `title`, hoặc format title khác kỳ vọng như "ĐIỀU 12" viết hoa, hoặc có dấu `:` khác vị trí).

**Acceptance criteria:**
- In ra tỉ lệ match thực tế của toàn bộ corpus (không chỉ 1 law đầu).
- Sau khi mở rộng regex/field parse đúng, `aid_map` phải có số entries xấp xỉ tổng số article thật trong corpus (hiện 245 là con số nghi ngờ quá thấp).
- Micro Law F1 trên 50 case phải > 0.

---

## Nhóm B — Di trú sang Llama-3.1-8B-Instruct qua NVIDIA NIM + LangChain `with_structured_output`

**Định hướng đã chốt:** triển khai **từng phần (hybrid)** — chỉ chuyển bước `predict_outcome` (quyết định nhãn + citation, ảnh hưởng trực tiếp 70%+10% điểm) sang NIM trước; giữ nguyên Qwen3.5-0.8B local cho `rewrite_query`/`decompose_query`/`case_digest` (không cần suy luận pháp lý sâu, tránh đốt quota NIM cho việc không cần thiết).

### B1. Cấu hình — `backend/config.py` + `backend/.env.example`

```python
# --- Generation backend switch (cho phép A/B so sánh với model cũ) ---
GENERATION_BACKEND = _env("GENERATION_BACKEND", "local")  # "local" | "nvidia_nim"
NVIDIA_API_KEY = _env("NVIDIA_API_KEY", "")
NVIDIA_NIM_MODEL_NAME = _env("NVIDIA_NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
NVIDIA_NIM_MAX_TOKENS = _env_int("NVIDIA_NIM_MAX_TOKENS", 1024)
```
Thêm dòng tương ứng vào `.env.example`. **KHÔNG** hardcode API key (đoạn code mẫu người dùng cung cấp có `api_key="$NVIDIA_API_KEY"` — đây là placeholder literal, không phải shell-expand tự động trong Python, phải đọc qua `os.getenv`/`config.py` như mọi key khác trong hệ thống).

### B2. Client wrapper — `backend/models.py`

```python
from functools import lru_cache
from langchain_nvidia_ai_endpoints import ChatNVIDIA

@lru_cache(maxsize=4)  # cache theo temperature vì ChatNVIDIA cố định temperature lúc khởi tạo
def _get_nim_client(temperature: float, max_tokens: int) -> ChatNVIDIA:
    if not config.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set — add it to .env (see README).")
    return ChatNVIDIA(
        model=config.NVIDIA_NIM_MODEL_NAME,
        api_key=config.NVIDIA_API_KEY,
        temperature=temperature,
        top_p=0.7,
        max_tokens=max_tokens,
    ).with_retry(stop_after_attempt=3)  # retry/backoff có sẵn của LangChain, thay _RateLimiter thủ công
```

### B3. Schema Pydantic thay thế JSON tự-parse — `backend/generation/generate.py`

```python
from pydantic import BaseModel, Field
from typing import Literal

class LawCitationOut(BaseModel):
    law_id: str
    aid: int

class VerdictOutput(BaseModel):
    prediction: Literal["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"]
    law_citations: list[LawCitationOut] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    accepted_ratio_estimate: float | None = Field(
        default=None,
        description="Chỉ dùng để log/QA hiệu chỉnh độ tự tin, KHÔNG override prediction",
    )


def predict_outcome_nim(case_query, law_chunks, case_digest) -> OutcomePrediction:
    system_prompt, user_prompt = build_prediction_prompt(case_query, law_chunks, case_digest)
    client = _get_nim_client(temperature=0.2, max_tokens=config.NVIDIA_NIM_MAX_TOKENS)
    structured = client.with_structured_output(VerdictOutput)

    try:
        result = structured.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        return _safe_default(f"NIM request failed: {e}")

    if result is None:
        # strict mode của with_structured_output: schema không dựng được -> None,
        # KHÔNG phải exception — phải check riêng, khác nhánh lỗi mạng ở trên.
        return _safe_default("NIM structured output returned None (schema validation failed)")

    allowed = allowed_citation_keys(law_chunks)
    kept, dropped = [], 0
    for c in result.law_citations:
        key = (c.law_id, c.aid)
        if key in allowed:
            kept.append(LawEvidenceItem(law_id=c.law_id, aid=c.aid))
        else:
            dropped += 1

    confidence = result.confidence
    reasoning = result.reasoning
    if not kept and confidence > _UNGROUNDED_CONFIDENCE_CEILING:
        reasoning = f"[confidence capped: no grounded law citation survived verification] {reasoning}"
        confidence = min(confidence, _UNGROUNDED_CONFIDENCE_CEILING)

    return OutcomePrediction(
        prediction=result.prediction,
        law_citations=kept,
        confidence=confidence,
        reasoning=reasoning,
        dropped_hallucinated_citations=dropped,
        is_fallback=False,
        fallback_reason=None,
    )
```

**Vì sao cách này tự động sửa cả A1 và A3 ở tầng gốc:** `prediction` giờ là `Literal` được ép bởi schema qua tool-calling — model buộc phải chọn đúng 1 trong 4 nhãn, không còn đường vòng qua `_label_from_ratio()`/ngưỡng `>0.99`. `law_citations` được model/inference engine tự đảm bảo đúng cấu trúc JSON (constrained decoding qua function-calling), loại bỏ hẳn lớp lỗi "Expecting ',' delimiter". `accepted_ratio_estimate` vẫn giữ để log/QA nhưng không còn quyền quyết định nhãn.

**Trong `pipeline.py`:** thêm nhánh rẽ theo `config.GENERATION_BACKEND`:
```python
from backend.generation.generate import predict_outcome, predict_outcome_nim

def _predict(case_query, law_chunks, case_digest):
    if config.GENERATION_BACKEND == "nvidia_nim":
        return predict_outcome_nim(case_query, law_chunks, case_digest)
    return predict_outcome(case_query, law_chunks, case_digest)
```

### B4. `requirements.txt` — thêm (không xoá gì bắt buộc)

```
langchain-core>=0.3
langchain-nvidia-ai-endpoints>=0.3
```
`torch`/`transformers`/`sentence-transformers` **vẫn giữ nguyên** — embedding, reranker, NER vẫn chạy local; chỉ bước sinh verdict cuối chuyển sang NIM.

### B5. Xử lý lỗi mạng/quota NIM

- `.with_retry(stop_after_attempt=3)` đã xử lý retry cơ bản có backoff.
- Cân nhắc thêm rate-limiter tương tự `case_api_client._RateLimiter` nếu API key ở tier miễn phí bị giới hạn request/phút — quan sát log NIM thực tế (429) trước khi quyết định có cần hay không, tránh đoán trước số liệu chưa kiểm chứng.
- Pre-flight check trong `test_all_backend.py` (giống check `ALQAC_TOKEN`/`BM25_INDEX_PATH` đã có): nếu `GENERATION_BACKEND=nvidia_nim` mà `NVIDIA_API_KEY` rỗng → thoát sớm với thông báo rõ ràng, không để chạy nửa chừng rồi crash hàng loạt.

### B6. Kế hoạch triển khai & đo lường tách bạch

1. Làm Nhóm A trước (đặc biệt A1) — vì đây là bug logic độc lập, rẻ, nên sửa ngay bất kể có di trú model hay không.
2. Cài B1–B4, chỉ bật `GENERATION_BACKEND=nvidia_nim` cho `predict_outcome`, giữ local cho phần còn lại.
3. `python -m test.test_all_backend -n 10` — kiểm tra không có lỗi 429/schema-None hàng loạt trước khi chạy full.
4. Chạy full 50 case, so sánh với 2 baseline đã có (0.260 gốc, 0.380 sau 3.1–3.4) để **tách riêng** đóng góp của: (a) sửa A1 một mình, (b) đổi model 8B qua NIM cộng thêm.
5. Nếu ổn định và ngân sách API cho phép, mở rộng pha 2: chuyển `case_digest`/`decompose_query` sang NIM.

---

## Nhóm C — Việc còn tồn đọng, ROI thấp hơn nhưng vẫn đáng làm

| # | Việc | File | Ghi chú |
|---|---|---|---|
| C1 | Giảm query fan-out để giảm phần tốc độ còn lại (~125s/case) | `hybrid_search.py`, `querry_transform.py` | Sau khi 3.1 đã xử lý overhead kết nối, phần lớn thời gian còn lại là do SỐ LƯỢNG query (rewrite ×5 + decomposition ×4 + vòng lặp evaluator), không phải do thiếu cache — cân nhắc giảm `n_variants` mặc định hoặc thắt điều kiện kích hoạt retrieval-evaluator |
| C2 | Tăng `FINAL_LAW_TOP_K` hoặc điều tra vì sao 9/50 case (~18%) bị hallucination-guard xoá sạch citation | `pipeline.collect_law_evidence`, `config.py` | Cho thấy retrieval đôi khi bỏ sót đúng điều luật cần thiết ngay từ khâu truy hồi, không chỉ ở khâu sinh câu trả lời |
| C3 | `config.API_HARD_CEILING_MULTIPLIER` là dead config — không hàm nào đọc lại (`test_all_backend.py._e_i()` tự hardcode `5 * budget_n`) | `config.py`, `test_all_backend.py` | Nên nối lại hoặc xoá để tránh gây hiểu nhầm khi đọc `.env.example` |
| C4 | `CHILD_CHUNK_MAX_TOKENS`/`PARENT_CHUNK_MAX_TOKENS` là dead config, `chunker.py` chỉ dùng `CHILD_MAX_CHARS` | `config.py`, `chunker.py` | Xoá hoặc nối lại cho nhất quán |
| C5 | `backend/generation/compress.py` không còn được `pipeline.py` gọi (vai trò đã chuyển sang `case_digest.py`) | `compress.py` | Quyết định: xoá hẳn hay giữ làm tuỳ chọn tương lai; nếu xoá, `llmlingua` trong `requirements.txt` cũng có thể bỏ |

---

## Nhóm D — Tài liệu cần cập nhật lại theo mã nguồn thực tế (đang dở dang)

Việc này đã bắt đầu ở phiên trước nhưng **chưa hoàn tất** v4:

- **`README.md`** — bản cập nhật chi tiết (kiến trúc thực tế, bảng đầy đủ biến môi trường, mục "mã nguồn còn tồn tại nhưng không còn được gọi") đã soạn xong nội dung nhưng ghi đè file bị lỗi `"File already exists"` — cần dùng `str_replace` hoặc `bash_tool` (`cat > path << 'EOF'`) để ghi đè thay vì `create_file`, sau đó bổ sung mục Nhóm B (di trú NIM) và cập nhật bảng "Vấn đề đã biết" theo đúng trạng thái ở mục 0 của tài liệu này.
- **`requirements.txt`** — chưa viết lại; cần thêm `numpy` (dùng trực tiếp trong `embed.py` nhưng chưa khai báo), ghi chú rõ `llmlingua`/`compress.py` hiện là code chết, và nếu làm Nhóm B thì thêm `langchain-core`/`langchain-nvidia-ai-endpoints`.
- **`docs/test_design.md`** — chưa viết lại; cần giữ nguyên phần A (spec chính thức của tổ chức thi, không tự ý diễn giải lại), nhưng bổ sung phần B mô tả rõ: (a) `backend/submission.py._validate_submission()` mirror đúng các rule nào, (b) giới hạn của `test_all_backend.py` (chỉ chạy được trên Public set vì cần `verdict_label`/`related_law_provisions`), (c) ghi chú rõ Micro Law F1 đo local hiện vẫn không đáng tin cho đến khi Nhóm A4 xong.

---

## Bảng tổng hợp ưu tiên cuối cùng (single backlog, sắp theo thứ tự nên làm)

| Thứ tự | Mục | Nhóm | Effort | Impact | Phụ thuộc |
|---|---|---|---|---|---|
| 1 | A1 — Hạ ngưỡng ratio `>0.99` xuống `>=0.9` | A | Rất thấp (1 dòng) | **Cao nhất** — mở khoá A_WIN ngay lập tức | Không |
| 2 | A2 — Bọc mã lỗi HTTP lạ (404) trong `case_api_client` | A | Thấp | Trung bình — tránh mất trắng cả case | Không |
| 3 | A3 — JSON-repair cho `law_citations` | A | Thấp | Trung bình — giảm ~5% fallback | Không (tạm thời, có thể bỏ nếu làm B) |
| 4 | A4 — Điều tra coverage `aid_map` | A | Trung bình | Cao — điều kiện để Micro Law F1 có ý nghĩa | Không |
| 5 | B1–B4 — Hạ tầng NIM (config, client, schema) | B | Trung bình | Cao (dài hạn) | Nên làm sau A1 để so sánh tách bạch |
| 6 | B6 — Chạy đo A/B theo kế hoạch | B | Thấp (chỉ chạy) | Xác nhận toàn bộ chuỗi thay đổi | Cần B1–B4 xong |
| 7 | D — Cập nhật README/requirements/test_design | D | Trung bình | Thấp về điểm số, cao về khả năng bảo trì | Nên làm sau khi A+B ổn định để tài liệu phản ánh đúng trạng thái cuối |
| 8 | C1–C5 — Dọn dẹp tốc độ + dead code | C | Thấp–Trung bình | Thấp–Trung bình | Không, có thể làm song song bất kỳ lúc nào |
