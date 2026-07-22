"""
Final prompt assembly for outcome prediction (design doc §4, §7.1;
system_adjustments_v3.md §5).

Enforces the "Grounding & Citation nghiêm ngặt" requirement:
  - The LLM must cite Điều/Khoản/Điểm for every legal claim.
  - The LLM must refuse / hedge if the retrieved context is insufficient.
  - Law-provision text is inserted VERBATIM (never compressed — see
    generation/compress.py) so no connective word can be silently dropped.
  - Case-evidence text is NOT passed in raw here anymore. It is pre-condensed
    by generation/case_digest.py into a short digest (system_adjustments_v3.md
    §5 — this was the biggest single contributor to the ~10k-token final
    prompt, and a <1B generation model degrades faster on long noisy context
    than the previously-planned 8B model would have).

IMPROVEMENT_PLAN.md §3.4 (ACCURACY fix, applied here): the system prompt now
also asks for a quantitative "accepted_ratio_estimate" (0.0-1.0) BEFORE the
categorical label, with fixed thresholds mapping ratio -> label, plus a
balanced 4-label few-shot block. Root cause being addressed: across the 50
public test cases, predictions never once landed on A_WIN or PARTIAL_B_WIN —
only PARTIAL_A_WIN (the categorical "safe middle" pick) and B_WIN (the
_safe_default() fallback label) ever appeared. A percentage estimate is a
different kind of judgement call than picking one of four discrete buckets
cold, and is intended to be less prone to collapsing onto a "safe-looking"
default under uncertainty. See generate.py for how the ratio is used to
derive (and, on disagreement, override) the categorical label.

Output contract: the model is asked to return a single JSON object so
`generation/generate.py` can parse it deterministically instead of
regex-scraping free-form prose.
"""
from __future__ import annotations

from backend.models import RetrievedChunk

SYSTEM_PROMPT = """Bạn là một hệ thống hỗ trợ dự đoán kết quả vụ án dân sự dựa trên pháp luật Việt Nam.

QUY TẮC BẮT BUỘC:
1. Chỉ được kết luận dựa trên các điều khoản pháp luật và bằng chứng vụ án được cung cấp bên dưới. KHÔNG được bịa thêm điều khoản, số hiệu văn bản, hoặc tình tiết không có trong ngữ cảnh.
2. Mọi luận điểm pháp lý PHẢI trích dẫn cụ thể (law_id, aid) từ danh sách "CÁC ĐIỀU LUẬT LIÊN QUAN" bên dưới. Không trích dẫn điều luật không có trong danh sách này.
3. Nếu ngữ cảnh được cung cấp không đủ căn cứ để kết luận chắc chắn, vẫn phải chọn nhãn khả dĩ nhất trong 4 nhãn nhưng phải hạ "confidence" xuống thấp (0.0-0.4) và nêu rõ trong "reasoning" rằng căn cứ còn hạn chế.
4. Nhãn dự đoán (prediction) phải là một trong: A_WIN, PARTIAL_A_WIN, PARTIAL_B_WIN, B_WIN.
   - A_WIN: tòa chấp nhận toàn bộ yêu cầu của nguyên đơn (bên A).
   - PARTIAL_A_WIN: tòa chấp nhận một phần, phần được chấp nhận > 50%.
   - PARTIAL_B_WIN: tòa chấp nhận một phần, phần được chấp nhận <= 50%.
   - B_WIN: tòa bác toàn bộ yêu cầu của nguyên đơn.
   Nếu vụ án có nhiều yêu cầu, chỉ tập trung vào yêu cầu chính (case_query).
5. TRƯỚC KHI chọn nhãn, hãy ước lượng "accepted_ratio_estimate": tỉ lệ (số thực 0.0-1.0) yêu cầu của nguyên đơn mà bạn cho rằng được tòa chấp nhận, dựa thuần túy trên bằng chứng và điều luật đã cho. Suy ra nhãn TỪ CHÍNH tỉ lệ này theo đúng bảng sau (không tự ý chọn nhãn khác với bảng):
   - ratio > 0.99         -> A_WIN
   - 0.5  < ratio <= 0.99 -> PARTIAL_A_WIN
   - 0.0  < ratio <= 0.5  -> PARTIAL_B_WIN
   - ratio == 0.0         -> B_WIN
   KHÔNG được mặc định chọn "PARTIAL_A_WIN" hoặc "B_WIN" chỉ vì đây là lựa chọn nghe "an toàn". Hãy ước lượng ratio một cách trung thực dựa trên chứng cứ thực tế — kể cả khi điều đó dẫn tới A_WIN hoặc PARTIAL_B_WIN.
6. Trả lời CHỈ bằng một đối tượng JSON hợp lệ theo đúng schema sau, không thêm văn bản nào khác, không dùng markdown code fence:
{
  "accepted_ratio_estimate": <float 0.0-1.0>,
  "prediction": "A_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN" | "B_WIN",
  "law_citations": [{"law_id": "...", "aid": <int>}, ...],
  "confidence": <float 0-1>,
  "reasoning": "giải thích ngắn gọn, có trích dẫn Điều/Khoản"
}

VÍ DỤ (chỉ minh hoạ định dạng và cách suy ra nhãn từ ratio, không phải nội dung thật):
- Bằng chứng + điều luật cho thấy nguyên đơn có đầy đủ căn cứ, bị đơn không phản bác được -> ratio=1.0 -> "prediction": "A_WIN"
- Tòa nhiều khả năng chấp nhận phần lớn yêu cầu (ví dụ ~70%) -> ratio=0.7 -> "prediction": "PARTIAL_A_WIN"
- Tòa nhiều khả năng chỉ chấp nhận một phần nhỏ yêu cầu (ví dụ ~30%) -> ratio=0.3 -> "prediction": "PARTIAL_B_WIN"
- Bằng chứng cho thấy bị đơn thắng hoàn toàn, không có căn cứ nào ủng hộ nguyên đơn -> ratio=0.0 -> "prediction": "B_WIN\""""


def _format_law_section(chunks: list[RetrievedChunk]) -> str:
    """Verbatim law text — NEVER pass through compress_auxiliary_text."""
    if not chunks:
        return "(Không có điều luật liên quan nào được truy hồi.)"
    lines = []
    for c in chunks:
        lines.append(f"- [{c.law_id} | Điều {c.aid}]\n{c.text.strip()}")
    return "\n\n".join(lines)


def build_prediction_prompt(
    case_query: str,
    law_chunks: list[RetrievedChunk],
    case_digest: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) ready for `backend.models.generate_text`.

    `case_digest` is the already-condensed case-fact summary produced by
    `generation/case_digest.build_case_digest` — this function no longer
    accepts raw `CaseEvidenceHit`s (see module docstring, system_adjustments_v3
    §5).
    """
    law_section = _format_law_section(law_chunks)

    user_prompt = f"""TÌNH HUỐNG VỤ ÁN (case_query):
{case_query.strip()}

TÓM TẮT BẰNG CHỨNG VỤ ÁN (đã được tổng hợp):
{case_digest.strip()}

CÁC ĐIỀU LUẬT LIÊN QUAN ĐÃ TRUY HỒI (law evidence, nguyên văn — chỉ được trích dẫn trong danh sách này):
{law_section}

Hãy đưa ra dự đoán kết quả vụ án theo đúng schema JSON đã quy định."""

    return SYSTEM_PROMPT, user_prompt


def allowed_citation_keys(law_chunks: list[RetrievedChunk]) -> set[tuple[str, int]]:
    """The closed set of (law_id, aid) pairs the model was actually shown —
    used by generate.py's verification pass to drop any hallucinated
    citation the model invents despite rule #2 in the system prompt."""
    return {(c.law_id, c.aid) for c in law_chunks}
