"""Streamlit chat UI for the labor-law RAG API.

Talks to the FastAPI backend over HTTP only (no direct import of `rag`), so it
runs unmodified whether the API is on localhost or the `api` service in
docker-compose. The sidebar exposes chunking strategy / retrieval mode /
reranker toggles so the same question can be re-run under different configs —
a live version of the ablation study.
"""

import os

import httpx
import streamlit as st
from refusal_labels import refusal_stage_label

API_URL = os.environ.get("API_URL", "http://localhost:8000")

STRATEGY_LABELS = {"structure": "依條文結構切分 (structure-aware)", "fixed": "固定長度切分 (fixed-size)"}
MODE_LABELS = {"hybrid": "Hybrid (BM25 + 向量)", "vector": "純向量 (BGE-M3)", "bm25": "純關鍵字 (BM25)"}

st.set_page_config(page_title="勞動法規 RAG 問答", page_icon="⚖️", layout="centered")
st.title("⚖️ 繁體中文勞動法規問答系統")
st.caption("知識庫:全國法規資料庫 15 部勞動法規（OGDL 開放授權）｜ Hybrid Search + Reranker + 引用來源")

with st.sidebar:
    st.subheader("檢索設定")
    strategy = st.selectbox("Chunking 策略", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get)
    mode = st.selectbox("檢索模式", list(MODE_LABELS), format_func=MODE_LABELS.get)
    use_reranker = st.checkbox("啟用 Reranker (bge-reranker-v2-m3)", value=True)
    st.divider()
    st.caption("調整設定後,下一個問題會用新設定重新檢索——可直接比較不同組合的效果與引用結果。")

    st.divider()
    try:
        health_response = httpx.get(f"{API_URL}/health", timeout=5.0)
        health_response.raise_for_status()
        health = health_response.json()
        st.success(f"後端就緒：{health['llm_provider']} ({health['generation_model']})")
        st.caption(
            f"structure: {health.get('collection_structure_points')} chunks ｜ "
            f"fixed: {health.get('collection_fixed_points')} chunks"
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        st.error(f"無法連線到 API ({API_URL})")

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"引用來源（{len(sources)}）"):
        for src in sources:
            st.markdown(f"**[{src['index']}] {src['doc']} {src['article']}**")
            st.caption(src["content"])


def render_debug(payload: dict) -> None:
    with st.expander("檢索細節（debug）"):
        st.json(
            {
                "strategy": payload["strategy"],
                "mode": payload["mode"],
                "use_reranker": payload["use_reranker"],
                "provider": f"{payload['provider']} ({payload['model']})",
                "refusal_stage": refusal_stage_label(
                    payload.get("refusal_stage"), refused=payload.get("refused", False)
                ),
                "retrieval_hits": payload["retrieval_hits"],
            }
        )


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("refused"):
            st.warning(msg["content"])
            st.caption(
                "拒答階段："
                + refusal_stage_label(
                    msg.get("payload", {}).get("refusal_stage"), refused=True
                )
            )
        else:
            st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("payload"):
            render_debug(msg["payload"])

question = st.chat_input("輸入你的勞動法規問題...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("檢索並生成答案中..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "strategy": strategy,
                        "mode": mode,
                        "use_reranker": use_reranker,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                st.error(f"API 呼叫失敗：{exc}")
                st.stop()

        if data["refused"]:
            st.warning(data["answer"])
            st.caption(
                "拒答階段："
                + refusal_stage_label(data.get("refusal_stage"), refused=True)
            )
        else:
            st.markdown(data["answer"])
        render_sources(data["sources"])
        render_debug(data)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "refused": data["refused"],
            "sources": data["sources"],
            "payload": data,
        }
    )
