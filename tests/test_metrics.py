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
