"""
Pre-retrieval query transformation (design doc §2.1, §2.3;
legalrag_adjustments.md §3).

- rewrite_query(): turns colloquial language into legal-register variants
  (3-5 paraphrases) so BM25/vector search hit the terms actually used in
  statutes.
- decompose_query(): asks the LLM to list distinct legal *aspects* that need
  looking up (e.g. the disputed legal relationship, contract-validity
  conditions, statute of limitations) as short standalone questions — NOT to
  draft hypothetical statute text.

HyDE (generate_hyde) has been REMOVED. It asked the LLM to draft a passage
"in the style of" a law provision or court finding and then embedded that
generated text for vector search. design_doc §2.3 already flagged the risk
(the LLM can produce legal-sounding but incorrect content that then steers
retrieval), and that risk gets worse, not better, on a much smaller
generation model. decompose_query replaces it: it only asks for short,
generic *aspect* questions ("what governs contract validity here?"), never
statute-styled prose, which keeps hallucinated legal content out of the
retrieval query entirely. See backend/retrieval/hybrid_search.py for how the
decomposed sub-queries are fused (weighted RRF, Judge-R1-style) with the
standard BM25/vector routes.
"""
from __future__ import annotations

from backend.models import generate_text

_REWRITE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Nhiệm vụ: viết lại câu hỏi/tình huống sau đây thành "
    "3 đến 5 câu hỏi tương đương, dùng thuật ngữ pháp lý chính xác thay cho "
    "ngôn ngữ đời thường (ví dụ: 'đánh nhau' -> 'hành vi cố ý gây thương tích', "
    "'lấy trộm' -> 'hành vi trộm cắp tài sản'). Mỗi câu hỏi trên một dòng, "
    "không đánh số, không giải thích thêm."
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Cho tình huống dưới đây, hãy liệt kê 3-4 khía cạnh pháp lý "
    "riêng biệt cần tra cứu (ví dụ: quan hệ pháp luật tranh chấp, điều kiện có hiệu lực "
    "của hợp đồng/giao dịch, thời hiệu khởi kiện, nghĩa vụ chứng minh). "
    "Mỗi khía cạnh viết thành MỘT câu hỏi ngắn, không đánh số, không giải thích thêm, "
    "KHÔNG được tự bịa nội dung điều luật cụ thể — chỉ nêu khía cạnh cần tra."
)


def rewrite_query(query: str, n_variants: int = 4, max_new_tokens: int = 256) -> list[str]:
    """Return `query` plus up to `n_variants` legal-register paraphrases.

    Falls back to just [query] if generation fails for any reason — a failed
    rewrite should never block retrieval entirely.
    """
    try:
        raw = generate_text(
            system_prompt=_REWRITE_SYSTEM_PROMPT,
            user_prompt=query,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
        )
    except Exception:
        return [query]

    variants = [line.strip("-• \t") for line in raw.splitlines() if line.strip()]
    variants = [v for v in variants if len(v) > 5][:n_variants]
    return [query] + variants if variants else [query]


def decompose_query(
    query: str,
    masked_query: str | None = None,
    n_subqueries: int = 4,
    max_new_tokens: int = 200,
) -> list[str]:
    """Replacement for generate_hyde(): decompose the case query into the
    distinct legal aspects that need to be looked up, as short standalone
    questions. Never generates hypothetical statute text (see module
    docstring for why HyDE was removed).

    Returns [] on failure or if generation produced nothing usable — callers
    should treat that as "skip the decomposition route", not as an error.
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
