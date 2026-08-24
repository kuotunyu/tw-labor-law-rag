from rag.ingestion.cleaner import clean_text, normalize_label


def test_newline_normalization():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_control_chars_removed_but_meaningful_chars_kept():
    assert clean_text("勞\x00工\x1f保﻿險") == "勞工保險"
    # Full-width digits/parens are meaningful in statutes — must survive.
    assert clean_text("（一）第１項") == "（一）第１項"


def test_trailing_whitespace_and_blank_lines():
    assert clean_text("第一項  \n\n\n\n第二項\t") == "第一項\n\n第二項"
    assert clean_text("條文　　\n內容") == "條文\n內容"  # full-width trailing space


def test_normalize_label():
    assert normalize_label("第  24  條") == "第 24 條"
    assert normalize_label(" 第 三 章 工資 ") == "第 三 章 工資"
