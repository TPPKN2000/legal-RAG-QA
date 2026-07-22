# LegalRAG — Đề xuất điều chỉnh V4

**Nguồn dữ liệu phân tích:** `test/test_all_backend.log` + `test/test_submission_backend.json`, chạy đầy đủ 50 case công khai (`ALQAC2026_public_test.json`, seed=42), commit hiện tại của `backend/`.

**Cách dùng tài liệu này:** Mỗi mục ở phần 3 là một unit-of-work độc lập, có: file cần sửa, đoạn code hiện tại (nguyên văn), chẩn đoán nguyên nhân, patch đề xuất, và tiêu chí chấp nhận (acceptance criteria) để tự kiểm tra sau khi sửa. Thực hiện theo đúng thứ tự ROI đã liệt kê — mục 3.1 và 3.2 không phụ thuộc lẫn nhau nên có thể làm song song; mục 3.3 và 3.4 nên làm sau vì cần số liệu sạch hơn từ 3.1/3.2 để đánh giá đúng tác động.

---

## 1. Số liệu tổng quan (baseline — trước khi sửa)

| Chỉ số | Giá trị | Ghi chú |
|---|---|---|
| OutcomeAccuracy | 0.260 (13/50) | Trọng số 70% trong FinalScore |
| Micro Law F1 (matches evaluation.md §2.6) | 0.000 (P=0.000, R=0.000) | Trọng số 10% |
| Avg Law F1 (macro, approx) | 0.000 | Cùng vấn đề như Micro F1 |
| Tổng API calls / case (avg) | 6.3 (313 calls / 50 case) | Ngân sách mặc định = 8/case |
| Thời gian xử lý trung bình/case | 197.0s | Không tính thời gian warm-up model (26.9s, chạy 1 lần) |
| Phân phối prediction | `PARTIAL_A_WIN`=35, `B_WIN`=15, `A_WIN`=0, `PARTIAL_B_WIN`=0 | **Không bao giờ** ra 2/4 nhãn |
| Phân phối gold | `PARTIAL_A_WIN`=19, `A_WIN`=16, `B_WIN`=10, `PARTIAL_B_WIN`=5 | Phân phối gold khá cân bằng, không giải thích được sự lệch của prediction |
| Estimated avg API efficiency E_i | 1.000 | Không phạt trong ước lượng local (dùng cùng fallback budget nên không phản ánh rủi ro thật) |
| Approx score (không tính Case Recall) | 0.70×0.260 + 0.10×0.000 = **0.182** | |

Ví dụ log lỗi API xuất hiện lặp lại (ảnh hưởng case_evidence rỗng/thiếu):
```
2026-07-21 16:08:55,460 [WARNING] backend.pipeline: case_api_client.retrieve failed for case=case_3241 query='...': /retrieve failed after 3 attempts: None
2026-07-21 17:17:18,356 [WARNING] backend.pipeline: case_api_client.retrieve failed for case=case_2978 query='...': /retrieve failed after 3 attempts: None
2026-07-21 16:59:48,686 [WARNING] backend.pipeline: case_api_client.retrieve failed for case=case_3079 query='...': /retrieve failed after 3 attempts: None
2026-07-21 17:02:36,257 [WARNING] backend.pipeline: case_api_client.retrieve failed for case=case_2705 query='...': /retrieve failed after 3 attempts: None
```

Ví dụ log cho thấy `api_calls` vượt `DEFAULT_MAX_API_CALLS_PER_CASE=8` (case_3241: `api_calls=9`, case_2978: `api_calls=10`, case_3079: `api_calls=9`) — liên quan trực tiếp đến mục 3.2.

Ví dụ block lặp lại trước MỌI lệnh gọi vector search (liên quan mục 3.1):
```
2026-07-21 15:51:21,018 [INFO] pinecone.client.indexes: Listing indexes
2026-07-21 15:51:22,757 [INFO] pinecone.client.indexes: Listing indexes  <- lặp lại 30+ lần/case
2026-07-21 15:51:24,273 [INFO] pinecone.index: Index client created for host ...
2026-07-21 15:51:24,308 [INFO] pinecone.index: Querying index with top_k=30
```

---

## 2. Chẩn đoán chi tiết (root cause)

### 2.1 Overhead khởi tạo Pinecone Index client (→ mục 3.1)
File: `backend/indexing/vector_store.py`

```python
@lru_cache(maxsize=1)
def _get_client():
    ...
    return Pinecone(api_key=config.PINECONE_API_KEY)

def ensure_index() -> None:
    pc = _get_client()
    existing = {i["name"] for i in pc.list_indexes()}   # <-- network round-trip
    ...

def _get_index():
    ensure_index()                                       # <-- gọi MỌI lần
    return _get_client().Index(config.PINECONE_INDEX_NAME)  # <-- tạo Index object MỌI lần

def query(text, top_k=..., law_id=None, require_active=True) -> list[RetrievedChunk]:
    index = _get_index()   # <-- không cache, chạy lại ensure_index() + tạo Index mới
    ...
```

`_get_client()` có `@lru_cache` nhưng `_get_index()` thì KHÔNG. Mỗi lần `query()` được gọi (BM25+vector chạy song song cho mỗi biến thể rewrite × mỗi sub-query decomposition × có thể nhân đôi nếu retrieval-evaluator loop kích hoạt) sẽ:
1. Gọi lại `pc.list_indexes()` — 1 network round-trip.
2. Gọi `pc.describe_index(...)` bên trong Pinecone SDK khi tạo `Index` object — thêm 1 round-trip.
3. Tạo lại `Index` client mới.

Với ~14–30 lượt gọi `vector_store.query()` mỗi case (`config.BM25_TOP_K`/`VECTOR_TOP_K`=30, `rewrite_query` mặc định trả tối đa 5 biến thể, `QUERY_DECOMPOSITION_MAX_SUBQUERIES`=4, cộng thêm vòng lặp retrieval-evaluator khi `score < RETRIEVAL_EVALUATOR_SCORE_THRESHOLD=0.75`), tổng round-trip dư thừa cộng dồn là nguồn chi phí chính trong 197s/case trung bình — tách biệt hoàn toàn với thời gian sinh văn bản của LLM.

### 2.2 Đếm nhầm API call khi retry (→ mục 3.2)
File: `backend/case_api_client.py`

```python
def retrieve(self, query: str, case_id: str) -> CaseEvidenceHit | None:
    ...
    for attempt in range(self.max_retries):
        self._limiter.wait()
        self._calls_per_case[case_id] += 1   # <-- TĂNG Ở MỌI VÒNG LẶP, kể cả retry
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            last_exc = e
            continue
        if resp.status_code == 200:
            ...
            return CaseEvidenceHit(...)
        if resp.status_code == 429:
            time.sleep(config.ALQAC_MIN_REQUEST_INTERVAL_SEC)
            continue          # <-- vòng lặp tiếp theo LẠI tăng _calls_per_case
        if resp.status_code == 503:
            time.sleep(1.0 * (attempt + 1))
            continue          # <-- tương tự
        ...
```

Một truy vấn logic (1 lời gọi `pipeline.collect_case_evidence` cho 1 query string) có thể chiếm 2-3 đơn vị trong bộ đếm `_calls_per_case` nếu API trả về 429/503/timeout tạm thời — những lỗi này KHÔNG phải do hệ thống gọi sai, nhưng vẫn bị tính vào ngân sách `c_i` dùng để tính `E_i` (API efficiency penalty, `docs/evaluation.md §2.4`). Bằng chứng trực tiếp: `case_3241` có `api_calls=9` dù `DEFAULT_MAX_API_CALLS_PER_CASE=8` — tức bộ đếm đã vượt ngân sách cấu hình chỉ vì retry.

Vì `docs/case_content_api_doc.md` ghi rõ giới hạn "1 request / 5 giây / team", và pipeline hiện đã có `_RateLimiter` xử lý đúng phần giãn cách — vấn đề duy nhất là **định nghĩa "1 call" trong bộ đếm nội bộ không khớp với "1 request" mà tổ chức thi tính**. Tổ chức thi (theo tài liệu) tính call theo request HTTP thực sự gửi đi — nếu vậy bug này thực ra khớp với cách họ tính, NHƯNG retry do lỗi mạng/503 là lãng phí ngân sách một cách không cần thiết (có thể tránh bằng cách retry ít hơn hoặc xử lý khác), nên vẫn cần sửa theo hướng giảm số lần retry vô ích, không phải chỉ sửa cách đếm.

### 2.3 Law F1 = 0.000 tuyệt đối — lệch namespace `aid` (→ mục 3.3)
File liên quan: `backend/ingestion/parser.py` (sinh `aid`), `test/test_all_backend.py` (parse gold + so khớp)

Gold parsing trong harness:
```python
def parse_gold_law_provisions(related_law_text: str) -> list[dict]:
    ...
    for line in related_law_text.strip().split("\n"):
        ...
        law_name, article_part = (p.strip() for p in line.split("|", 1))
        m = re.search(r"Điều\s+(\d+)", article_part)
        if m:
            provisions.append({"law_name": law_name, "article_num": int(m.group(1))})
    return provisions

def compute_law_f1(predicted: list[dict], gold_provisions: list[dict]) -> tuple[float, float, float]:
    pred_aids = {int(p["aid"]) for p in predicted}
    gold_aids = {g["article_num"] for g in gold_provisions}
    ...
```

`gold_aids` là các số "Điều N" nhỏ (vài chục–vài trăm, kiểu văn bản luật Việt Nam thông thường). Nhưng `pred_aids` lấy từ `record_dict["law_evidence"]`, mà `aid` này được sinh trong `parser.load_law_corpus()`:
```python
aid = art.get("aid")
if aid is None:
    aid = art.get("id")
```
Xem `test_submission_backend.json`, các giá trị `aid` thực tế là: `50882, 50910, 50908, 14343, 53082, 53354, 50893, 50691, 53197, 50769, ...` — đây rõ ràng **không phải số "Điều N"** kiểu văn bản luật thông thường, mà giống ID tuần tự/hash nội bộ chạy xuyên suốt toàn bộ corpus (global sequential/composite ID). Hai tập `pred_aids` và `gold_aids` sống trong hai namespace hoàn toàn khác nhau về độ lớn và ngữ nghĩa → giao luôn rỗng → Precision, Recall, F1 luôn bằng 0 (trừ trường hợp cả hai đều rỗng, cho F1=1.0 theo code, nhưng thực tế trong log KHÔNG case nào đạt điều này dù một số case có `law_evidence=[]` — cần kiểm tra thêm liệu `gold_provisions` của các case đó có rỗng hay không để xác nhận).

Đây là **lỗi cấu trúc phép đo**, không phải "retrieval kém" — không thể sửa bằng cách cải thiện retrieval/rerank, phải sửa cách map gold.

### 2.4 Dự đoán sụp về 2/4 nhãn (→ mục 3.4)
File liên quan: `backend/generation/generate.py`, `backend/generation/prompt_builder.py`

```python
def _safe_default(reason: str) -> OutcomePrediction:
    return OutcomePrediction(
        prediction="B_WIN",   # <-- LUÔN LUÔN B_WIN khi có lỗi generation/parsing
        law_citations=[],
        confidence=0.0,
        reasoning=f"[fallback] {reason}",
    )
```
`_safe_default` được gọi khi: (a) `generate_text()` raise exception, hoặc (b) `_extract_json()` không tìm được JSON hợp lệ, hoặc (c) `prediction not in config.VALID_PREDICTIONS`. **Không có counter nào đếm tần suất rơi vào nhánh này** — hiện tại không thể phân biệt "model tự tin chọn B_WIN" với "pipeline crash và ép về B_WIN".

Đồng thời, `prompt_builder.SYSTEM_PROMPT` rule #3 chỉ hướng dẫn hạ `confidence` khi ngữ cảnh yếu, **không có cơ chế hoặc ví dụ nào ép/khuyến khích model chọn nhãn cực trị (`A_WIN`, `PARTIAL_B_WIN`)** — không có few-shot minh hoạ 4 nhãn cân bằng trong prompt. Với model rất nhỏ (Qwen3.5-0.8B), xu hướng tự nhiên khi không chắc là hội tụ về nhãn "an toàn ở giữa" (`PARTIAL_A_WIN`) khi có ngữ cảnh, và về `B_WIN` khi lỗi/parse fail — khớp chính xác với phân phối quan sát được (35× `PARTIAL_A_WIN`, 15× `B_WIN`, 0× 2 nhãn còn lại).

---

## 3. Danh sách sửa lỗi ưu tiên theo ROI

### 3.1. [SPEED — ROI cao nhất, effort thấp nhất] Cache Pinecone Index client

**File:** `backend/indexing/vector_store.py`

**Patch:**
```python
@lru_cache(maxsize=1)
def _get_index():
    """Cache the Index client — Pinecone SDK does a describe_index() network
    call when constructing Index(), so re-creating it on every query() call
    (previous behavior) added a network round-trip per BM25/vector query.
    """
    ensure_index()
    return _get_client().Index(config.PINECONE_INDEX_NAME)
```
Xoá định nghĩa `_get_index()` cũ (không có `@lru_cache`), thay bằng bản trên. Không cần đổi chữ ký gọi ở `query()`, `upsert_chunks()`, `delete_namespace()` — các hàm này đã gọi `_get_index()` sẵn.

**Lưu ý phụ:** Nếu sau này cần rebuild index / đổi namespace trong cùng process (vd. test suite gọi `delete_namespace()` rồi `ensure_index()` lại), cần expose một hàm `_get_index.cache_clear()` để invalidate cache — thêm helper `def reset_index_cache(): _get_index.cache_clear()` nếu cần.

**Acceptance criteria:**
- Chạy `python -m test.test_all_backend -n 3`, đếm số dòng log `"Listing indexes"` — phải giảm xuống còn **đúng 1 lần cho cả run** (thay vì 1 lần/query).
- Thời gian trung bình/case giảm rõ rệt so với baseline 197s (kỳ vọng giảm mạnh vì đây là phần lớn round-trip dư thừa quan sát được trong log).
- Kết quả prediction/law_evidence không đổi so với trước khi sửa (đây là thay đổi thuần về caching, không ảnh hưởng logic truy hồi).

---

### 3.2. [BUDGET INTEGRITY — ROI cao, effort thấp] Sửa đếm API call khi retry

**File:** `backend/case_api_client.py`

**Vấn đề cụ thể:** biến đếm tăng ở mọi vòng lặp `for attempt in range(self.max_retries)`, kể cả khi request đó là do lỗi 429/503/timeout (không phải do query mới).

**Patch đề xuất (đếm 1 lần cho mỗi *query logic*, không đếm theo attempt):**
```python
def retrieve(self, query: str, case_id: str) -> CaseEvidenceHit | None:
    if not self.token:
        raise CaseAPIError(
            "ALQAC_TOKEN is not set. Add it to your .env file (see README §Configuration)."
        )

    url = f"{self.base_url}/retrieve"
    headers = {"X-API-Key": self.token, "Content-Type": "application/json"}
    payload = {"query": query, "case_id": case_id}

    # Count exactly ONE call for this logical query, regardless of how many
    # transient-error retries it takes underneath — retries due to 429/503/
    # network hiccups are not additional "queries" the caller issued, and
    # must not inflate the API-efficiency budget c_i (docs/evaluation.md §2.4).
    self._calls_per_case[case_id] += 1

    last_exc: Exception | None = None
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
        resp.raise_for_status()

    raise CaseAPIError(f"/retrieve failed after {self.max_retries} attempts: {last_exc}")
```
Thay đổi cốt lõi: dòng `self._calls_per_case[case_id] += 1` được **chuyển ra ngoài vòng lặp `for attempt`**, chạy đúng 1 lần/lời gọi `retrieve()`.

**Cân nhắc bổ sung (không bắt buộc nhưng nên làm cùng lúc):** nếu tổ chức thi tính "call" theo số request HTTP thực sự gửi đi (kể cả retry) chứ không theo query logic, patch trên sẽ đánh giá thấp `c_i` thực tế khi nộp bài thật. Cần xác nhận với `docs/case_content_api_doc.md`/tổ chức thi cách họ đếm trước khi chốt — nếu không rõ, ít nhất nên **giảm số lần retry gây lãng phí** (vd. giảm `max_retries` mặc định từ 3 xuống 2, hoặc không retry ở 429 nếu limiter đã đảm bảo giãn cách đúng 5s) thay vì chỉ sửa cách đếm.

**Acceptance criteria:**
- Viết unit test mock `requests.post` trả về 429 hai lần rồi 200 ở lần thứ ba; gọi `client.retrieve(...)`; assert `client.calls_made(case_id) == 1` (không phải 3).
- Chạy lại `test_all_backend.py -n 10`, xác nhận không còn case nào có `api_calls > DEFAULT_MAX_API_CALLS_PER_CASE` trừ khi thực sự có ≥9 truy vấn logic riêng biệt được phát ra bởi `collect_case_evidence`.

---

### 3.3. [MEASUREMENT INTEGRITY — ROI cao nhưng effort trung bình] Sửa lệch namespace `aid` giữa gold và prediction

**File:** `test/test_all_backend.py` (chỗ đo), và cần kiểm tra `backend/ingestion/parser.py` / `data/corpus_law_pub.json` (nguồn `aid`)

**Bước 1 — Xác nhận giả thuyết (bắt buộc làm trước khi sửa code):**
```python
# Chạy 1 lần, không sửa code, chỉ để xác nhận
import json
corpus = json.load(open("data/corpus_law_pub.json", encoding="utf-8"))
sample_law = corpus[0] if isinstance(corpus, list) else corpus["laws"][0]
for art in sample_law.get("articles", sample_law.get("content", []))[:10]:
    print(art.get("aid"), art.get("id"), art.get("title"))
```
So sánh giá trị in ra với các con số nhỏ trong `title` (vd. "Điều 12"). Nếu `aid`/`id` KHÔNG khớp với số trong `title`, xác nhận giả thuyết ở mục 2.3 là đúng — corpus dùng ID nội bộ, không phải article number.

**Bước 2a — Nếu xác nhận đúng giả thuyết:** Harness đo cần một bảng ánh xạ thật `(law_name hoặc law_id, article_num) -> aid`, lấy trực tiếp từ `corpus_law_pub.json` thay vì suy diễn:
```python
def build_article_num_to_aid_map(corpus_path) -> dict[tuple[str, int], int]:
    """Map (law_id, article_num_parsed_from_title) -> real aid, built once from
    the actual corpus, so gold provisions (which only give law name + Điều N)
    can be resolved to the same aid namespace the pipeline predicts in."""
    docs = load_law_corpus(corpus_path)
    mapping = {}
    for doc in docs:
        for art in doc.articles:
            m = re.search(r"Điều\s+(\d+)", art.title)
            if m:
                mapping[(doc.law_id, int(m.group(1)))] = art.aid
    return mapping
```
Sau đó `parse_gold_law_provisions` cần trả về cả `law_name` lẫn `article_num`, và bước so khớp trong `compute_law_f1` / micro F1 phải join qua bảng `mapping` này (khớp gần đúng theo `law_name` vì gold chỉ có tên luật, không có `law_id` chuẩn — cần thêm bước fuzzy-match tên luật ↔ `law_id`, hoặc yêu cầu tổ chức thi cấp bảng gold chính thức có `law_id`).

**Bước 2b — Nếu KHÔNG xác nhận giả thuyết** (tức `aid` thực ra đúng là số Điều nhưng bị lỗi ở bước khác): kiểm tra `chunker.chunk_article()` xem `aid` có bị ghi đè bởi biến khác trong lúc build chunk hay không (vd. nhầm `parent_id` hash vào `aid`). Trong trường hợp này sửa tại `backend/ingestion/chunker.py`/`parser.py` thay vì tại harness đo.

**Acceptance criteria:**
- In ra tối thiểu 5 cặp `(pred_aid, gold_article_num)` đã match thành công sau khi sửa, xác nhận bằng mắt rằng chúng cùng trỏ đến 1 điều luật thật.
- Micro Law F1 trên tập 50 case phải > 0 (không cần cao, chỉ cần khác 0 để chứng minh phép đo đã hoạt động).
- Ghi rõ trong code/README rằng đây vẫn là **xấp xỉ cục bộ** (public test set không cấp `law_id` chuẩn cho gold), không phải công thức chấm điểm thật của tổ chức thi.

---

### 3.4. [ACCURACY — ROI cao nhất về điểm số nhưng effort cao nhất] Sửa sụp nhãn dự đoán

**File:** `backend/generation/generate.py`, `backend/generation/prompt_builder.py`, `backend/pipeline.py`

**Bước 1 — Instrumentation bắt buộc trước khi thử bất kỳ fix nào (để biết đang sửa đúng vấn đề):**

Trong `backend/generation/generate.py`, thêm phân loại rõ ràng lý do fallback thay vì gộp chung vào 1 string `reasoning`:
```python
@dataclass
class OutcomePrediction:
    prediction: Prediction
    law_citations: list[LawEvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    dropped_hallucinated_citations: int = 0
    is_fallback: bool = False          # <-- MỚI: true nếu đến từ _safe_default
    fallback_reason: str | None = None  # <-- MỚI


def _safe_default(reason: str) -> OutcomePrediction:
    return OutcomePrediction(
        prediction="B_WIN",
        law_citations=[],
        confidence=0.0,
        reasoning=f"[fallback] {reason}",
        is_fallback=True,
        fallback_reason=reason,
    )
```
Trong `test/test_all_backend.py`, thêm đếm và log tỉ lệ fallback:
```python
n_fallback = sum(1 for r in results if r.get("is_fallback"))
log.info("Fallback rate (crash -> forced B_WIN): %d/%d (%.1f%%)", n_fallback, n, 100 * n_fallback / n)
```
Và trong `pipeline.process_case`, truyền `is_fallback`/`fallback_reason` từ `outcome` vào `SubmissionRecord` hoặc log riêng (không nhất thiết đưa vào submission chính thức vì đó không thuộc schema thi, nhưng cần lộ ra được cho harness đo nội bộ).

**Kết quả mong đợi của bước 1:** một con số cụ thể — vd. "8/50 case (16%) là crash-fallback ép về B_WIN, còn lại 7/50 case B_WIN là model tự tin chọn". Con số này quyết định bước 2 nên tập trung vào đâu:
- Nếu tỉ lệ fallback cao (>20%) → ưu tiên sửa robustness của `_extract_json`/`generate_text` (model 0.8B có thể sinh JSON lỗi hoặc bị cắt do `max_new_tokens` không đủ, hoặc `<think>` leak).
- Nếu tỉ lệ fallback thấp → vấn đề nằm ở chính khả năng phân loại 4 nhãn của model, cần sửa prompt/few-shot hoặc đổi model.

**Bước 2a — Nếu do fallback/parsing (áp dụng ngay, effort thấp):**
- Tăng `GENERATION_MAX_NEW_TOKENS_DEFAULT` (hiện 500) nếu log cho thấy output bị cắt trước khi có JSON.
- Trong `_extract_json`, log nguyên văn `raw` khi thất bại (hiện chỉ raise `ValueError` với 200 ký tự đầu) để xem model đang sinh ra gì — có thể `<think>` leak dù đã có guard trong `models.generate_text`.

**Bước 2b — Nếu do model thật sự không chọn nhãn cực trị (effort trung bình):**
Sửa `backend/generation/prompt_builder.py`, thêm few-shot ngắn cân bằng 4 nhãn vào `SYSTEM_PROMPT`, ví dụ chèn trước "Trả lời CHỈ bằng...":
```python
SYSTEM_PROMPT = """... (giữ nguyên phần đầu) ...

VÍ DỤ (chỉ minh hoạ định dạng, không phải nội dung thật):
- Nếu bằng chứng cho thấy nguyên đơn được chấp nhận 100% yêu cầu và có căn cứ luật rõ ràng -> "prediction": "A_WIN"
- Nếu bị đơn thắng hoàn toàn, không có căn cứ nào ủng hộ nguyên đơn -> "prediction": "B_WIN"
- Nếu tòa chấp nhận một phần > 50% yêu cầu -> "prediction": "PARTIAL_A_WIN"
- Nếu tòa chỉ chấp nhận một phần <= 50% yêu cầu -> "prediction": "PARTIAL_B_WIN"
KHÔNG được mặc định chọn PARTIAL_A_WIN hoặc B_WIN chỉ vì đây là lựa chọn "an toàn" — hãy đánh giá % yêu cầu được chấp nhận dựa trên chứng cứ thực tế trước khi chọn nhãn.

... (giữ nguyên phần schema JSON còn lại) ..."""
```
- Cân nhắc thêm một field trung gian bắt model tự ước lượng % trước khi map nhãn (vd. thêm `"accepted_ratio_estimate": <float 0-1>` vào schema JSON, rồi validate logic: `ratio > 0.99 -> A_WIN`, `0.5 < ratio <= 0.99 -> PARTIAL_A_WIN`, `0 < ratio <= 0.5 -> PARTIAL_B_WIN`, `ratio == 0 -> B_WIN`) — cách này ép model suy luận định lượng thay vì nhảy thẳng vào nhãn categorical, dễ giảm sụp nhãn hơn.

**Bước 2c — Nếu model 0.8B không đủ khả năng dù đã sửa prompt (effort cao, cần hạ tầng):**
- Thử `GENERATION_MODEL_NAME` lớn hơn (Qwen3.5-1.8B-Instruct hoặc tương đương) nếu GPU cho phép, giữ nguyên toàn bộ pipeline retrieval — đây là thay đổi cấu hình (`config.py`/`.env`), không phải thay đổi code logic.

**Acceptance criteria:**
- Sau bước 1: có báo cáo tỉ lệ fallback cụ thể trong log của `test_all_backend.py`.
- Sau bước 2 (bất kỳ nhánh nào áp dụng): chạy lại 50 case, xác nhận phân phối prediction xuất hiện **cả 4 nhãn** (không nhất thiết cân bằng như gold, nhưng phải > 0 lần mỗi nhãn), và OutcomeAccuracy không thấp hơn baseline 0.260.

---

## 4. Tổng kết mức độ ưu tiên (tóm tắt 1 dòng/mục)

| # | Mục | Loại | Effort | Impact kỳ vọng |
|---|---|---|---|---|
| 3.1 | Cache Pinecone Index client | Speed | Rất thấp (1 dòng) | Giảm mạnh 197s/case |
| 3.2 | Sửa đếm API call khi retry | Budget integrity | Thấp | Tránh đốt oan `E_i` khi chấm điểm thật |
| 3.3 | Sửa lệch namespace `aid` (gold vs pred) | Measurement integrity | Trung bình | Micro Law F1 từ luôn-0 thành có ý nghĩa đo lường |
| 3.4 | Sửa sụp nhãn dự đoán | Accuracy | Trung bình–Cao | Tăng OutcomeAccuracy (thành phần 70% điểm) |
