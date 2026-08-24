from rag.indexing.tokenizer import tokenize


def test_tokenize_drops_punctuation_and_stopwords():
    tokens = tokenize("勞工結婚可以請幾天婚假?")
    assert "?" not in tokens
    assert "的" not in tokens
    assert tokens  # non-empty


def test_custom_legal_terms_stay_whole():
    """Terms from dict/legal_terms.txt must not be split, regardless of the base dict."""
    assert "資遣費" in tokenize("資遣費怎麼計算?")
    assert "加班費" in tokenize("雇主延長工作時間的加班費怎麼算?")
    assert "特別休假" in tokenize("特別休假有幾天?")
    assert "勞動基準法" in tokenize("勞動基準法規定了什麼?")


def test_tokenize_is_idempotent_across_calls():
    a = tokenize("勞工保險條例")
    b = tokenize("勞工保險條例")
    assert a == b
