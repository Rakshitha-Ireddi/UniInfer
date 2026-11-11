from uniinfer.metrics import anls, levenshtein, mc_accuracy, normalize_answer, vqa_accuracy


def test_normalize_strips_articles_punct_case():
    assert normalize_answer("  The Dakota!  ") == "dakota"


def test_vqa_accuracy_counts_annotators():
    answers = ["dakota"] * 2 + ["other"] * 8
    assert vqa_accuracy("Dakota", answers) == 2 / 3
    assert vqa_accuracy("dakota", ["dakota"] * 3 + ["x"] * 7) == 1.0
    assert vqa_accuracy("nope", answers) == 0.0


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3


def test_anls_threshold():
    assert anls("0.28", ["0.28"]) == 1.0
    assert 0.5 < anls("0.29", ["0.28"]) < 1.0
    assert anls("completely different", ["0.28"]) == 0.0
    assert anls("x", ["y", "x"]) == 1.0


def test_mc_accuracy():
    assert mc_accuracy("B", ["B"]) == 1.0
    assert mc_accuracy("B. Because", ["B"]) == 1.0
    assert mc_accuracy("A", ["B"]) == 0.0
    assert mc_accuracy("none", ["B"]) == 0.0


def test_relaxed_accuracy_numeric_tolerance():
    from uniinfer.metrics import relaxed_accuracy

    assert relaxed_accuracy("12.4", ["12"]) == 1.0      # 3.3% off: within 5%
    assert relaxed_accuracy("15", ["12"]) == 0.0        # 25% off
    assert relaxed_accuracy("12.6", ["12"]) == 1.0      # exactly 5% off: boundary is inclusive
    assert relaxed_accuracy("0", ["0"]) == 1.0
    assert relaxed_accuracy("0.1", ["0"]) == 0.0


def test_relaxed_accuracy_formatting():
    from uniinfer.metrics import relaxed_accuracy

    assert relaxed_accuracy("Yes", ["yes"]) == 1.0
    assert relaxed_accuracy("yes.", ["Yes"]) == 1.0
    assert relaxed_accuracy("45%", ["45"]) == 1.0
    assert relaxed_accuracy("1,234", ["1234"]) == 1.0
    assert relaxed_accuracy("$12", ["12"]) == 1.0
    assert relaxed_accuracy("Lamb", ["Corn"]) == 0.0
    assert relaxed_accuracy("12 apples", ["12"]) == 0.0  # non-numeric prediction: exact match required


def test_scorer_registered():
    from uniinfer.data import DATASETS, MAX_NEW_TOKENS
    from uniinfer.metrics import SCORERS

    assert set(DATASETS) == set(SCORERS) == set(MAX_NEW_TOKENS)


def test_content_id_is_stable_and_sensitive():
    from PIL import Image

    from uniinfer.data import content_id

    a = Image.new("RGB", (8, 8), "white")
    b = Image.new("RGB", (8, 8), "black")
    assert content_id("q", a) == content_id("q", a.copy())
    assert content_id("q", a) != content_id("q", b) != content_id("r", a)
    assert len(content_id("q", a)) == 12
