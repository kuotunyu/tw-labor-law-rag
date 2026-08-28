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
from api_client import actual_generation_metadata, fetch_models, submit_query
from refusal_labels import refusal_stage_label

API_URL = os.environ.get("API_URL", "http://localhost:8000")

STRATEGY_LABELS = {"structure": "依條文結構切分 (structure-aware)", "fixed": "固定長度切分 (fixed-size)"}
MODE_LABELS = {"hybrid": "Hybrid (BM25 + 向量)", "vector": "純向量 (BGE-M3)", "bm25": "純關鍵字 (BM25)"}

st.set_page_config(page_title="勞動法規 RAG 問答", page_icon="⚖️", layout="centered")
st.title("⚖️ 繁體中文勞動法規問答系統")
st.caption("知識庫:全國法規資料庫 15 部勞動法規（OGDL 開放授權）｜ Hybrid Search + Reranker + 引用來源")


@st.cache_data(ttl=60)
def get_model_catalog(api_url: str) -> dict:
    """Cache the public, configured choices without retaining query data."""
    return fetch_models(api_url)


try:
    model_catalog = get_model_catalog(API_URL)
    provider_records = model_catalog.get("providers", [])
    if not isinstance(provider_records, list):
        provider_records = []
except (httpx.HTTPError, ValueError, TypeError):
    provider_records = []
    model_catalog = {}

provider_records = [
    record
    for record in provider_records
    if isinstance(record, dict)
    and isinstance(record.get("provider"), str)
    and isinstance(record.get("model"), str)
]
provider_models = {record["provider"]: record["model"] for record in provider_records}
available_providers = list(provider_models)
default_provider = model_catalog.get("default_provider")
if default_provider not in provider_models:
    default_provider = available_providers[0] if available_providers else None


def provider_label(provider: str) -> str:
    return f"{provider}（{provider_models[provider]}）"


def render_generation_status(payload: dict) -> None:
    requested_provider = payload.get("requested_provider")

    if requested_provider in provider_models:
        st.caption(f"指定模型：{provider_label(requested_provider)}")
    if not payload.get("generation_called", True):
        st.info("此題在檢索階段拒答，未呼叫生成模型。")
        return
    actual_metadata = actual_generation_metadata(payload)
    if actual_metadata:
        provider, model = actual_metadata
        st.caption(f"實際作答模型：{provider}（{model}）")
    if payload.get("fallback_used"):
        st.warning("原模型暫時無法使用，已切換備援模型。")


with st.sidebar:
    st.subheader("檢索設定")
    strategy = st.selectbox("Chunking 策略", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get)
    mode = st.selectbox("檢索模式", list(MODE_LABELS), format_func=MODE_LABELS.get)
    use_reranker = st.checkbox("啟用 Reranker (bge-reranker-v2-m3)", value=True)
    st.divider()
    st.caption("調整設定後,下一個問題會用新設定重新檢索——可直接比較不同組合的效果與引用結果。")

    st.divider()
    st.subheader("回答模型")
    if available_providers:
        selected_provider = st.selectbox(
            "回答模型",
            available_providers,
            index=available_providers.index(default_provider),
            format_func=provider_label,
            label_visibility="collapsed",
        )
    else:
        selected_provider = None
        st.warning("目前沒有可用的回答模型，請稍後再試。")

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
    actual_metadata = actual_generation_metadata(payload)
    actual_model = (
        f"{actual_metadata[0]}（{actual_metadata[1]}）"
        if actual_metadata
        else "未呼叫生成模型"
    )
    with st.expander("檢索細節（debug）"):
        st.json(
            {
                "strategy": payload["strategy"],
                "mode": payload["mode"],
                "use_reranker": payload["use_reranker"],
                "provider": actual_model,
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
            render_generation_status(msg["payload"])
            render_debug(msg["payload"])

question = st.chat_input("輸入你的勞動法規問題...", disabled=selected_provider is None)
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("檢索並生成答案中..."):
            try:
                data = submit_query(
                    API_URL,
                    {
                        "question": question,
                        "strategy": strategy,
                        "mode": mode,
                        "use_reranker": use_reranker,
                        "provider": selected_provider,
                    },
                )
            except (httpx.HTTPError, ValueError, TypeError):
                st.error("目前無法取得回答，請稍後再試。")
                st.stop()

        if data["refused"]:
            st.warning(data["answer"])
            st.caption(
                "拒答階段："
                + refusal_stage_label(data.get("refusal_stage"), refused=True)
            )
        else:
            st.markdown(data["answer"])
        render_generation_status(data)
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
