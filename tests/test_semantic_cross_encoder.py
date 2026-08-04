"""Cross-encoder similarity kernel for DocumentSemantics."""

from __future__ import annotations

import numpy as np
import pytest

from linguistic_diversity import DocumentSemantics
from linguistic_diversity.diversities.semantic import cross_encoder_similarities

STS = "cross-encoder/stsb-roberta-large"


@pytest.mark.slow
def test_identical_documents_give_diversity_one() -> None:
    """The axiom a cross-encoder breaks unless identical strings are forced to 1.

    Cross-encoders score a sentence against itself at ~0.96, which would report 1.15
    effective species for four identical documents.
    """
    metric = DocumentSemantics({"cross_encoder": STS, "verbose": False})
    assert metric(["The cat sat on the mat."] * 4) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_unrelated_documents_need_no_floor() -> None:
    """Unrelated text scores near zero, which is why the floor is not applied."""
    metric = DocumentSemantics({"cross_encoder": STS, "verbose": False})
    docs = [
        "The stock market closed higher on Tuesday.",
        "She baked a loaf of sourdough bread.",
        "The telescope detected a distant galaxy.",
        "He repaired the bicycle's rear brake.",
    ]
    features, _ = metric.extract_features(docs)
    Z = metric.calculate_similarities(features)
    off_diagonal = Z[~np.eye(len(docs), dtype=bool)]
    assert off_diagonal.max() < 0.25
    assert metric(docs) > 3.6


@pytest.mark.slow
def test_matrix_is_symmetric_bounded_and_unit_diagonal() -> None:
    metric = DocumentSemantics({"cross_encoder": STS, "verbose": False})
    docs = ["A cat sat.", "A dog barked loudly.", "The kettle boiled over."]
    features, _ = metric.extract_features(docs)
    Z = metric.calculate_similarities(features)
    assert np.allclose(Z, Z.T)
    assert np.all((Z >= 0.0) & (Z <= 1.0))
    assert np.allclose(np.diag(Z), 1.0)


@pytest.mark.slow
def test_paraphrases_score_below_distinct_documents() -> None:
    metric = DocumentSemantics({"cross_encoder": STS, "verbose": False})
    paraphrases = [
        "The cat sat on the mat.",
        "A feline rested upon the rug.",
        "On the mat, the cat was sitting.",
    ]
    distinct = [
        "The cat sat on the mat.",
        "Bond yields fell after the central bank spoke.",
        "The telescope detected a distant galaxy.",
    ]
    assert metric(paraphrases) < metric(distinct)


@pytest.mark.slow
def test_refuses_corpora_above_the_pair_budget() -> None:
    metric = DocumentSemantics(
        {"cross_encoder": STS, "cross_encoder_max_docs": 4, "verbose": False}
    )
    with pytest.raises(ValueError, match="cross_encoder_max_docs"):
        metric([f"doc {i}" for i in range(6)])


@pytest.mark.slow
def test_requires_extract_features_first() -> None:
    metric = DocumentSemantics({"cross_encoder": STS, "verbose": False})
    with pytest.raises(ValueError, match="extract_features"):
        metric.calculate_similarities(np.zeros((3, 8)))


@pytest.mark.slow
def test_nli_model_is_detected_and_uses_entailment() -> None:
    """An NLI checkpoint is auto-detected from its label map, not assumed."""
    metric = DocumentSemantics(
        {"cross_encoder": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "verbose": False}
    )
    assert metric(["The cat sat on the mat."] * 3) == pytest.approx(1.0, abs=1e-6)
    assert (
        metric(["The cat sat on the mat.", "Bond yields fell.", "A distant galaxy drifted."]) > 2.5
    )


@pytest.mark.slow
def test_rejects_a_classifier_with_no_entailment_label() -> None:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    model.config.num_labels = 2
    model.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    with pytest.raises(ValueError, match="no 'entailment' label"):
        cross_encoder_similarities(["a", "b"], model)
