"""
Final prompt assembly for outcome prediction (design doc §4, §7.1).

Enforces the "Grounding & Citation nghiêm ngặt" requirement:
  - The LLM must cite Điều/Khoản/Điểm for every legal claim.
  - The LLM must refuse / hedge if the retrieved context is insufficient.
  - Law-provision text is inserted VERBATIM (never compressed — see
    generation/compress.py) so no connective word can be silently dropped.
  - Case-evidence text may be compressed (auxiliary, not statute text).

Output contract: the model is asked to return a single JSON object so
`generation/generate.py` can parse it deterministically instead of
regex-scraping free-form prose.
"""
from __future__ import annotations

from backend import config
from backend.models import CaseEvidenceHit, RetrievedChunk

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
5. Trả lời CHỈ bằng một đối tượng JSON hợp lệ theo đúng schema sau, không thêm văn bản nào khác, không dùng markdown code fence:
{
  "prediction": "A_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN" | "B_WIN",
  "law_citations": [{"law_id": "...", "aid": <int>}, ...],
  "confidence": <float 0-1>,
  "reasoning": "giải thích ngắn gọn, có trích dẫn Điều/Khoản"
}"""


def _format_law_section(chunks: list[RetrievedChunk]) -> str:
    """Verbatim law text — NEVER pass through compress_auxiliary_text."""
    if not chunks:
        return "(Không có điều luật liên quan nào được truy hồi.)"
    lines = []
    for c in chunks:
        lines.append(f"- [{c.law_id} | Điều {c.aid}]\n{c.text.strip()}")
    return "\n\n".join(lines)


def _format_case_evidence_section(hits: list[CaseEvidenceHit], compressed_texts: list[str] | None = None) -> str:
    if not hits:
        return "(Không có bằng chứng vụ án nào được truy hồi.)"
    texts = compressed_texts if compressed_texts is not None else [h.text for h in hits]
    lines = []
    for hit, text in zip(hits, texts):
        lines.append(f"- [{hit.chunk_id}] {text.strip()}")
    return "\n\n".join(lines)


def build_prediction_prompt(
    case_query: str,
    law_chunks: list[RetrievedChunk],
    case_evidence_hits: list[CaseEvidenceHit],
    compressed_evidence_texts: list[str] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) ready for `backend.models.generate_text`."""
    law_section = _format_law_section(law_chunks)
    evidence_section = _format_case_evidence_section(case_evidence_hits, compressed_evidence_texts)

    user_prompt = f"""TÌNH HUỐNG VỤ ÁN (case_query):
{case_query.strip()}

BẰNG CHỨNG VỤ ÁN ĐÃ TRUY HỒI (case evidence):
{evidence_section}

CÁC ĐIỀU LUẬT LIÊN QUAN ĐÃ TRUY HỒI (law evidence, nguyên văn — chỉ được trích dẫn trong danh sách này):
{law_section}

Hãy đưa ra dự đoán kết quả vụ án theo đúng schema JSON đã quy định."""

    return SYSTEM_PROMPT, user_prompt


def allowed_citation_keys(law_chunks: list[RetrievedChunk]) -> set[tuple[str, int]]:
    """The closed set of (law_id, aid) pairs the model was actually shown —
    used by generate.py's verification pass to drop any hallucinated
    citation the model invents despite rule #2 in the system prompt."""
    return {(c.law_id, c.aid) for c in law_chunks}
