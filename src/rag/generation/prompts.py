"""Prompt templates: citation format and the honest-refusal instruction."""

from __future__ import annotations

from rag.models import RetrievedChunk

# The LLM must emit this exact phrase when it cannot answer from the given
# context — the answerer matches on it verbatim, so refusal detection is a
# plain substring check rather than a fuzzy heuristic.
REFUSAL_PHRASE = "知識庫中沒有相關資訊"

SYSTEM_PROMPT = f"""你是台灣勞動法規的問答助理,只能根據使用者提供的法規條文回答問題。

規則:
1. 只能使用提供的條文內容作答,不得使用條文以外的知識或臆測。
2. 回答中每個論點都要用 [數字] 標註對應的條文編號,對應提供內容的順序,例如 [1]、[2]。
3. 如果提供的條文不足以回答問題,或問題與提供的條文無關,必須明確回答「{REFUSAL_PHRASE}」,不要試圖用其他知識拼湊答案。
4. 用簡潔、口語化的繁體中文回答,不要逐字複誦條文,而是用白話解釋重點。"""


def build_context_block(hits: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] {hit.citation}\n{hit.payload['content']}" for i, hit in enumerate(hits, start=1)
    )


def build_user_prompt(question: str, hits: list[RetrievedChunk]) -> str:
    context = build_context_block(hits) if hits else "(無檢索結果)"
    return f"""以下是知識庫中檢索到的相關法規條文:

{context}

使用者問題:{question}"""
