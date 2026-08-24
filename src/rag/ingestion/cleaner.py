"""Text normalization for ingested documents.

Deliberately conservative for legal text: Unicode NFC + whitespace cleanup only.
Characters that carry meaning in statutes (full-width digits and parentheses,
historical wording like 「左列」) are never rewritten — NFKC would silently
change how citations render.
"""

from __future__ import annotations

import re
import unicodedata

# C0/C1 control characters (except tab \x09 and newline \x0a) plus the BOM,
# built programmatically to keep the source file free of literal control bytes.
_CONTROL_CODEPOINTS = (
    list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20)) + [0x7F, 0xFEFF]
)
_CONTROL_CHARS = re.compile("[" + re.escape("".join(map(chr, _CONTROL_CODEPOINTS))) + "]")
_TRAILING_WS = re.compile(r"[ \t　]+$", flags=re.MULTILINE)
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def normalize_label(label: str) -> str:
    """Collapse internal whitespace in article/chapter labels: '第  24  條' -> '第 24 條'."""
    return re.sub(r"\s+", " ", label.strip())
