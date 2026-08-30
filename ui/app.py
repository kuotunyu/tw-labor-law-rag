"""Streamlit chat UI for the labor-law RAG API.

Talks to the FastAPI backend over HTTP only (no direct import of `rag`), so it
runs unmodified whether the API is on localhost or the `api` service in
docker-compose. Expert retrieval controls remain available behind progressive
disclosure so first-time visitors can focus on the secure BYOK question flow.
"""

import os
from urllib.parse import urlparse

import httpx
import streamlit as st

try:
    from ui import _bootstrap as _ui_bootstrap
except ModuleNotFoundError:  # Streamlit starts with ui/ as the import root.
    import _bootstrap as _ui_bootstrap

_ui_bootstrap.ensure_import_roots()

from ui.api_client import (  # noqa: E402
    ApiRequestError,
    actual_generation_metadata,
    fetch_models,
    fetch_session,
    requested_provider_for_display,
    submit_query,
)
from ui.content import (  # noqa: E402
    BYOK_PRIVACY_POINTS,
    EXAMPLE_QUESTIONS,
    KNOWLEDGE_BASE,
)
from ui.refusal_labels import refusal_stage_label  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")

STRATEGY_LABELS = {"structure": "依條文結構切分 (structure-aware)", "fixed": "固定長度切分 (fixed-size)"}
MODE_LABELS = {"hybrid": "Hybrid (BM25 + 向量)", "vector": "純向量 (BGE-M3)", "bm25": "純關鍵字 (BM25)"}
PROVIDER_NAMES = {"gemini": "Gemini", "openai": "OpenAI"}

st.set_page_config(page_title="勞動法規 RAG 問答", page_icon="⚖️", layout="centered")
st.markdown(
    """
    <style>
    :root {
      --law-ink: #20242c;
      --law-muted: #667085;
      --law-paper: #fbfaf7;
      --law-line: #d8d2c5;
      --law-accent: #a43b32;
      --law-accent-soft: #f5e9e6;
    }
    .stApp { background: var(--law-paper); color: var(--law-ink); }
    [data-testid="stHeader"] { background: rgba(251, 250, 247, .92); }
    [data-testid="stSidebar"] { border-right: 1px solid var(--law-line); }
    .block-container { max-width: 980px; padding-top: 2.25rem; }
    .law-eyebrow {
      color: var(--law-accent);
      font-size: .78rem;
      font-weight: 700;
      letter-spacing: .12em;
      margin-bottom: .35rem;
      text-transform: uppercase;
    }
    .law-rule { border-top: 1px solid var(--law-line); margin: 1.25rem 0; }
    div.stButton > button { min-height: 2.75rem; border-color: var(--law-line); }
    div.stButton > button:hover { border-color: var(--law-accent); color: var(--law-accent); }
    @media (max-width: 700px) {
      .block-container { padding: 1rem .85rem 5rem; }
    }
    </style>
    <div class="law-eyebrow">Evidence-first legal retrieval</div>
    """,
    unsafe_allow_html=True,
)
st.title("⚖️ 繁體中文勞動法規問答")
st.caption(
    f"知識庫快照：{KNOWLEDGE_BASE.snapshot_date}　·　"
    f"{KNOWLEDGE_BASE.laws} 部法規／{KNOWLEDGE_BASE.articles} 條非刪除條文　·　"
    "Hybrid Search + Reranker + 法源引用"
)
st.caption("資料源為全國法規資料庫（OGDL 授權）；本系統僅供法規檢索與技術展示，不是法律意見。")


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
    with st.expander("進階比較設定", expanded=False):
        strategy = st.selectbox(
            "Chunking 策略",
            list(STRATEGY_LABELS),
            format_func=STRATEGY_LABELS.get,
        )
        mode = st.selectbox(
            "檢索模式",
            list(MODE_LABELS),
            format_func=MODE_LABELS.get,
        )
        use_reranker = st.checkbox(
            "啟用 Reranker (bge-reranker-v2-m3)", value=True
        )
        st.caption("這些選項用於比較消融設定；一般問答可維持預設值。")
    st.divider()
    st.caption("調整後，下一個問題會用新設定重新檢索；歷史回答仍保留原設定與引用。")

selected_provider = None
visitor_key = None
with st.container(border=True):
    st.subheader("三步開始問答")
    st.caption("① 選模型　　② 貼上自己的 API Key　　③ 選範例或直接提問")
    st.caption("本站不使用站長的模型額度，也不在伺服器保存訪客金鑰。")

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
        key_column, clear_column = st.columns([5, 2], vertical_alignment="bottom")
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
            st.info("API Key 已填入，但尚未向模型供應商驗證；第一次成功送出後才代表可用。")
        else:
            st.info(f"請輸入 {provider_name} API Key，再選擇範例或提出問題。")

        st.caption("　•　".join(BYOK_PRIVACY_POINTS))
        query_limit = st.session_state.get("demo_query_limit")
        if query_limit:
            st.caption(f"每個展示工作階段最多 {query_limit} 次查詢。")
        if not st.session_state.get("demo_session"):
            st.warning("目前無法建立展示工作階段，請稍後重新整理頁面。")
    elif selected_provider is not None:
        st.success("模型服務已由部署環境設定完成，可以開始問答。")

if "history" not in st.session_state:
    st.session_state.history = []

byok_ready = bool(
    not requires_api_key
    or (
        isinstance(visitor_key, str)
        and visitor_key.strip()
        and st.session_state.get("demo_session")
    )
)

st.subheader("試一個代表性問題")
st.caption("範例不使用預先寫好的答案；點擊後會走與自行提問完全相同的檢索與生成流程。")
pending_question = None
example_columns = st.columns(2)
for index, item in enumerate(EXAMPLE_QUESTIONS):
    with example_columns[index % len(example_columns)]:
        if st.button(
            item.title,
            key=f"example-{item.id}",
            help=f"{item.category}｜{item.question}",
            disabled=selected_provider is None or not byok_ready,
            use_container_width=True,
        ):
            pending_question = item.question

st.markdown('<div class="law-rule"></div>', unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"引用來源（{len(sources)}）"):
        for src in sources:
            st.markdown(f"**[{src['index']}] {src['doc']} {src['article']}**")
            st.caption(src["content"])
            last_amended = str(src.get("last_amended", "")).strip()
            if len(last_amended) == 8 and last_amended.isdigit():
                last_amended = (
                    f"{last_amended[:4]}-{last_amended[4:6]}-{last_amended[6:]}"
                )
            if last_amended:
                st.caption(f"最新異動：{last_amended}")
            source_url = str(src.get("source_url", "")).strip()
            parsed_url = urlparse(source_url)
            if parsed_url.scheme == "https" and parsed_url.hostname == "law.moj.gov.tw":
                st.markdown(f"[全國法規資料庫]({source_url})")


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

typed_question = st.chat_input(
    "輸入你的勞動法規問題...",
    disabled=selected_provider is None or not byok_ready,
)
question = typed_question or pending_question
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
