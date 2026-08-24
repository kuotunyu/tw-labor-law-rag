"""Traditional-Chinese tokenizer for BM25, built on jieba.

jieba's bundled ``dict.txt`` is simplified-Chinese oriented and contains almost
none of our domain's traditional multi-character terms (e.g. 資遣費, 特別休假
are entirely absent — verified by grepping the installed dict). Two layers fix
this:

1. ``dict.txt.big``, the traditional-Chinese dictionary maintained in the
   jieba project itself (MIT license), fetched once and cached locally.
2. A small hand-curated list of labor-law terms (``dict/legal_terms.txt``)
   that even the big dict tends to over-split, loaded as a jieba user
   dictionary so they always segment as single tokens.

Falls back to jieba's default dictionary if the big-dict download fails
(e.g. offline dev environment) — segmentation quality degrades but the
pipeline keeps working.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

with warnings.catch_warnings():
    # jieba 0.42.1 imports the deprecated pkg_resources API. The dependency
    # lock keeps setuptools on the last compatible line; suppress only that
    # known import-time warning so CLI/API smoke checks stay actionable.
    warnings.filterwarnings(
        "ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning
    )
    import jieba

from rag.config import PROJECT_ROOT

DICT_BIG_URL = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big"
LEGAL_TERMS_PATH = Path(__file__).resolve().parent / "dict" / "legal_terms.txt"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "storage" / "dict"

_STOPWORDS = frozenset(
    "的 之 於 及 或 與 等 為 而 且 亦 並 但 即 者 也 得 應 予 所 其 此 該 依 按 使 令".split()
)
_TOKEN_PATTERN = re.compile(r"[一-鿿]+|[A-Za-z0-9]+")

_initialized_from: Path | None = None


def ensure_traditional_dict(cache_dir: Path) -> Path | None:
    """Download dict.txt.big into ``cache_dir`` if not already cached; return its path, or None on failure."""
    path = cache_dir / "dict.txt.big"
    if path.exists():
        return path
    try:
        import httpx

        cache_dir.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(DICT_BIG_URL, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
        return path
    except Exception as exc:  # network unavailable, etc. — caller falls back to the default dict
        print(f"[tokenizer] warning: could not fetch traditional dict.txt.big ({exc}); "
              f"falling back to jieba's default (simplified-oriented) dictionary")
        return None


def init_tokenizer(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Idempotent: re-initializing with the same cache_dir is a no-op."""
    global _initialized_from
    if _initialized_from == cache_dir:
        return

    dict_path = ensure_traditional_dict(cache_dir)
    if dict_path is not None:
        jieba.set_dictionary(str(dict_path))
    if LEGAL_TERMS_PATH.exists():
        jieba.load_userdict(str(LEGAL_TERMS_PATH))
    jieba.initialize()
    _initialized_from = cache_dir


def tokenize(text: str) -> list[str]:
    """Segment ``text`` into BM25-ready tokens (punctuation and stopwords dropped)."""
    if _initialized_from is None:
        init_tokenizer()
    tokens = jieba.cut(text)
    return [t for t in tokens if t not in _STOPWORDS and _TOKEN_PATTERN.fullmatch(t)]
