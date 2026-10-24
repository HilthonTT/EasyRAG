from easyrag.evaluation import exact_match, hit_rate, token_f1


def test_exact_match_normalizes() -> None:
    assert exact_match("The Paris.", ["paris"]) == 1.0
    assert exact_match("Berlin", ["paris"]) == 0.0


def test_token_f1_partial_overlap() -> None:
    assert 0.0 < token_f1("Paris France", ["Paris"]) < 1.0


def test_hit_rate() -> None:
    assert hit_rate(["a", "b"], ["b"]) == 1.0
    assert hit_rate(["a"], ["c"]) == 0.0
