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

from ui.api_client import (
    ApiRequestError,
    actual_generation_metadata,
    fetch_models,
    fetch_session,
    requested_provider_for_display,
    submit_query,
)
from ui.refusal_labels import refusal_stage_label

API_URL = os.environ.get("API_URL", "http://localhost:8000")

STRATEGY_LABELS = {"structure": "依條文結構切分 (structure-aware)", "fixed": "固定長度切分 (fixed-size)"}
MODE_LABELS = {"hybrid": "Hybrid (BM25 + 向量)", "vector": "純向量 (BGE-M3)", "bm25": "純關鍵字 (BM25)"}
PROVIDER_NAMES = {"gemini": "Gemini", "openai": "OpenAI"}

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
requires_api_key = model_catalog.get("requires_api_key") is True

if requires_api_key and "demo_session" not in st.session_state:
    try:
        session = fetch_session(API_URL)
    except (httpx.HTTPError, ValueError, TypeError):
        st.session_state.demo_session = None
        st.session_state.demo_query_limit = None
    else:
        st.session_state.demo_session = session["token"]
        st.session_state.demo_query_limit = session["query_limit"]


def provider_label(provider: str) -> str:
    return f"{PROVIDER_NAMES.get(provider, provider)} · {provider_models[provider]}"


def clear_provider_key() -> None:
    st.session_state["visitor_provider_key"] = ""


def byok_error_message(status_code: int) -> str:
    return {
        400: "輸入內容不符合公開展示的限制，請縮短問題或檢查設定。",
        401: "API Key 或展示工作階段無效，請確認 Key 後再試。",
        429: "目前已達展示額度或同時使用上限，請稍後再試。",
        502: "模型服務目前無法完成回答，請稍後再試。",
        503: "檢索服務尚未就緒，請稍後再試。",
        504: "模型回應超時，請稍後再試。",
    }.get(status_code, "目前無法取得回答，請稍後再試。")


def render_generation_status(payload: dict) -> None:
    requested_provider = requested_provider_for_display(payload)

    if requested_provider in provider_models:
        st.caption(f"指定模型：{provider_label(requested_provider)}")
    elif requested_provider:
        st.caption(f"指定模型：{requested_provider}")
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

selected_provider = None
visitor_key = None
with st.container(border=True):
    st.subheader("🔐 開始安全問答")
    st.caption("選擇模型並貼上你自己的 API Key；本站不使用站長的模型額度。")

    if available_providers:
        selected_provider = st.segmented_control(
            "回答模型",
            available_providers,
            default=default_provider,
            format_func=provider_label,
            selection_mode="single",
            key="selected_provider",
        )
    else:
        st.warning("目前沒有可用的回答模型，請稍後再試。")

    if requires_api_key and selected_provider is not None:
        if st.session_state.get("provider_key_provider") != selected_provider:
            st.session_state["visitor_provider_key"] = ""
            st.session_state["provider_key_provider"] = selected_provider

        provider_name = PROVIDER_NAMES.get(selected_provider, selected_provider)
        key_column, clear_column = st.columns([4, 1], vertical_alignment="bottom")
        with key_column:
            visitor_key = st.text_input(
                f"{provider_name} API Key",
                type="password",
                placeholder=f"貼上 {provider_name} API Key（輸入內容會隱藏）",
                key="visitor_provider_key",
            )
        with clear_column:
            st.button(
                "清除 API Key",
                on_click=clear_provider_key,
                disabled=not bool(visitor_key),
                use_container_width=True,
            )

        if visitor_key.strip():
            st.success("API Key 已填入，可以開始問答。")
        else:
            st.info(f"請輸入 {provider_name} API Key，再到下方提出問題。")

        st.caption(
            "只保留在目前瀏覽器工作階段　•　不寫入檔案或聊天紀錄　•　"
            "模型費用由 API Key 持有人承擔"
        )
        query_limit = st.session_state.get("demo_query_limit")
        if query_limit:
            st.caption(f"每個展示工作階段最多 {query_limit} 次查詢。")
        if not st.session_state.get("demo_session"):
            st.warning("目前無法建立展示工作階段，請稍後重新整理頁面。")
    elif selected_provider is not None:
        st.success("模型服務已由部署環境設定完成，可以開始問答。")

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

byok_ready = bool(
    not requires_api_key
    or (
        isinstance(visitor_key, str)
        and visitor_key.strip()
        and st.session_state.get("demo_session")
    )
)
question = st.chat_input(
    "輸入你的勞動法規問題...",
    disabled=selected_provider is None or not byok_ready,
)
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
                    api_key=visitor_key if requires_api_key else None,
                    session_token=(
                        st.session_state.get("demo_session")
                        if requires_api_key
                        else None
                    ),
                )
            except ApiRequestError as exc:
                st.error(byok_error_message(exc.status_code))
                st.stop()
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
