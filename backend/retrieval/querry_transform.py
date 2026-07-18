"""
Pre-retrieval query transformation (design doc §2.1, §2.3).

- rewrite_query(): turns colloquial language into legal-register variants
  (3-5 paraphrases) so BM25/vector search hit the terms actually used in
  statutes.
- generate_hyde(): asks the LLM to draft a hypothetical answer/provision
  text, then embeds *that* for vector search — text-to-text matches the
  dense embedding space better than question-to-text.

Both are logged by the caller (see pipeline.py) since HyDE in particular can
hallucinate legal-sounding but incorrect content; hybrid search with BM25
is the safety net, not this module.
"""
from __future__ import annotations

import re

from backend.models import generate_text

_REWRITE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Nhiệm vụ: viết lại câu hỏi/tình huống sau đây thành "
    "3 đến 5 câu hỏi tương đương, dùng thuật ngữ pháp lý chính xác thay cho "
    "ngôn ngữ đời thường (ví dụ: 'đánh nhau' -> 'hành vi cố ý gây thương tích', "
    "'lấy trộm' -> 'hành vi trộm cắp tài sản'). Mỗi câu hỏi trên một dòng, "
    "không đánh số, không giải thích thêm."
)

_HYDE_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Với tình huống pháp lý sau, hãy viết một đoạn văn "
    "ngắn (3-5 câu) mô phỏng văn phong một điều khoản pháp luật hoặc phần "
    "nhận định của tòa án liên quan đến tình huống này. Đây CHỈ dùng để hỗ trợ "
    "tìm kiếm, không phải câu trả lời cuối cùng, nên không cần chính xác tuyệt "
    "đối, chỉ cần đúng văn phong và các thuật ngữ pháp lý liên quan."
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


def generate_hyde(query: str, max_new_tokens: int = 200) -> str | None:
    """Generate a hypothetical legal passage for HyDE-style embedding search.

    Returns None on failure so callers can gracefully skip the HyDE branch.
    """
    try:
        text = generate_text(
            system_prompt=_HYDE_SYSTEM_PROMPT,
            user_prompt=query,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    except Exception:
        return None
