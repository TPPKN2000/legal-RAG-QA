"""
Case-fact digest step (design doc §7.1 follow-up; legalrag_adjustments.md
§5).

Why this exists: the ~10k-token final generation prompt did not come from
HyDE or query rewriting (those only ever see the short case_query). It came
from `prompt_builder.build_prediction_prompt` stuffing the *entire* set of
retrieved case-evidence hits plus the *entire* verbatim law-chunk text into
one single LLM call. That is both expensive and a poor fit for a <1B
generation model, which degrades faster on long, noisy context than an 8B
model would.

The fix splits generation into two smaller calls:

    case evidence hits (many, long)  -->  [LLM #1: this module]  --> digest (~150-250 words)
    case_query + digest + law_chunks (verbatim)  -->  [LLM #2: generate.py]  --> JSON verdict

Only the digest step touches raw case-evidence text, and it is explicitly
instructed not to add or infer anything beyond what's given — it is a
compression step, not a reasoning step. Legal reasoning and citation still
happen in the (verbatim-law-grounded) verdict call in generate.py.

Note: `case_evidence` reported in the final submission is NOT limited to the
hits used here — pipeline.py still submits the full retrieved set for
Case-Recall scoring purposes. Only the *digest input* is capped (see
config.TOP_N_EVIDENCE_FOR_DIGEST) to keep token usage bounded.
"""
from __future__ import annotations

from backend import config
from backend.models import generate_text

_DIGEST_SYSTEM_PROMPT = (
    "Bạn là trợ lý tóm tắt hồ sơ vụ án. Từ các đoạn bằng chứng vụ án dưới đây, hãy viết "
    "một đoạn TÓM TẮT NGẮN (tối đa 150 từ) nêu: các bên, yêu cầu của nguyên đơn, "
    "lập luận/bằng chứng chính của mỗi bên. KHÔNG suy đoán, KHÔNG thêm thông tin ngoài "
    "các đoạn được cung cấp. Nếu bằng chứng trống, trả lời đúng câu: "
    "'(Không có bằng chứng vụ án được truy hồi.)'"
)

_NO_EVIDENCE_DIGEST = "(Không có bằng chứng vụ án được truy hồi.)"


def build_case_digest(case_query: str, evidence_texts: list[str]) -> str:
    """Condense `evidence_texts` (already score-ranked/truncated by the
    caller — see config.TOP_N_EVIDENCE_FOR_DIGEST) into a short digest.

    Falls back to a hard-truncated concatenation of the raw texts on
    generation failure, so a digest-step error degrades quality rather than
    crashing the whole case (mirrors the fallback philosophy already used by
    query rewriting / decomposition in querry_transform.py).
    """
    if not evidence_texts:
        return _NO_EVIDENCE_DIGEST

    joined = "\n\n".join(f"- {t.strip()}" for t in evidence_texts if t.strip())
    if not joined:
        return _NO_EVIDENCE_DIGEST

    user_prompt = f"TÌNH HUỐNG:\n{case_query.strip()}\n\nBẰNG CHỨNG:\n{joined}"
    try:
        digest = generate_text(
            system_prompt=_DIGEST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=config.CASE_DIGEST_MAX_NEW_TOKENS,
            temperature=0.2,
        ).strip()
        return digest or _NO_EVIDENCE_DIGEST
    except Exception:
        # Digest generation failing must never block the whole case — fall
        # back to a hard-capped raw concatenation instead of raising.
        return joined[:800]
