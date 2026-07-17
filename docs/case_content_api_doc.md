# Retrieval API
During the evidence-retrieval phase, teams query case content through this API instead of reading the full case files. Each query returns the single most relevant content segment, which you collect as case_evidence to ground your outcome prediction. Report cited law provisions as { `law_id`, `aid` } pairs in `law_evidence`, where aid is the article's aid from the law corpus.
## Authentication
Every request must include your team's secret token in the X-API-Key header — the same token the organizers issued you for leaderboard submissions. Requests without a valid token are rejected with 403.
```env
X-API-Key: alqac_xxxxxxxxxxxxxxxxxxxxxxxx
```
## POST /retrieve
Retrieve the top-ranked evidence segment for a query within one case.

Request body (JSON):
- query: The Vietnamese search query — keywords describing the evidence you are looking for.
- case_id: The target case, e.g. case_1087_0037. Must be a case in the official public/private test set.

Response 200 — top-1 segment:
```json
{
  "results": [
    {
      "score": 0.886,
      "text": "Người có quyền lợi nghĩa vụ liên quan: ...",
      "chunk_id": "case_1087_0037_chunk_2"
    }
  ]
}
```
- chunk_id — the segment id to record in your submission's case_evidence.
- score — BM25 relevance score (higher is more relevant).
- Exactly one segment is returned per call; issue multiple queries to gather more evidence.
## Examples
CURL:
```curl
curl -X POST https://alqac-api.ngrok.pro/retrieve \
  -H "X-API-Key: $ALQAC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "tranh chấp quyền sử dụng đất", "case_id": "case_1087_0037"}'
```
Python:
```python
import requests

resp = requests.post(
    "https://alqac-api.ngrok.pro/retrieve",
    headers={"X-API-Key": "ALQAC_TOKEN"},
    json={
        "query": "tranh chấp quyền sử dụng đất",
        "case_id": "case_1087_0037",
    },
    timeout=30,
)
resp.raise_for_status()
for hit in resp.json()["results"]:
    print(hit["chunk_id"], round(hit["score"], 3), hit["text"][:120])
```
## Rate limits & errors
The API is limited to 1 request every 5 seconds per team. Pace your requests accordingly; exceeding the limit returns 429.
- 200: Success — results array returned.
- 403: Missing or invalid X-API-Key.
- 422: Malformed request (missing query or case_id).
- 429: Rate limit exceeded — wait 5 seconds and retry.
- 503: Team database temporarily unavailable.
## Bad Retrieval
The number of API calls per case feeds the 20% Penalized Case Recall component: case-evidence recall is multiplied by an API-efficiency factor that gives full credit up to 2·n calls and decays to zero at 5·n (where n is the number of segments in the case). Retrieve thoroughly, but economically — see the scoring rules for the full formula.