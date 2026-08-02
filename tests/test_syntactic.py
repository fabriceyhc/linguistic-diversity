"""Tests for syntactic diversity metrics."""

import pytest

from linguistic_diversity.diversities.syntactic import (
    KARATECLUB_AVAILABLE,
    ConstituencyParse,
    DependencyParse,
)

try:
    import benepar  # noqa: F401

    BENEPAR_AVAILABLE = True
except ImportError:
    BENEPAR_AVAILABLE = False

# karateclub pins numpy<1.23, which conflicts with this package's numpy>=1.24, so it
# cannot be installed alongside the library. The ldp/feather paths are skipped rather
# than failed so a clean install still yields a green suite.
requires_karateclub = pytest.mark.skipif(
    not KARATECLUB_AVAILABLE, reason="karateclub not installed (pins numpy<1.23)"
)
requires_benepar = pytest.mark.skipif(
    not BENEPAR_AVAILABLE, reason="benepar not installed (pip install '.[syntactic]')"
)


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "A fast red dog runs quickly",
        "The cat sleeps",
    ]


class TestDependencyParse:
    """Tests for DependencyParse metric."""

    @requires_karateclub
    @pytest.mark.slow
    def test_basic_diversity_ldp(self, sample_corpus):
        """Test basic dependency parse diversity with LDP."""
        metric = DependencyParse({"similarity_type": "ldp"})
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity > 0
        assert diversity < 10  # Reasonable range

    @requires_karateclub
    @pytest.mark.slow
    def test_basic_diversity_feather(self, sample_corpus):
        """Test basic dependency parse diversity with Feather."""
        metric = DependencyParse({"similarity_type": "feather"})
        diversity = metric(sample_corpus)

        assert diversity > 0
        assert diversity < 10

    @pytest.mark.slow
    def test_tree_edit_distance(self):
        """Test tree edit distance similarity."""
        # Small corpus for edit distance (slow)
        corpus = ["The cat sat", "A dog ran"]

        metric = DependencyParse({"similarity_type": "tree_edit_distance"})
        diversity = metric(corpus)

        assert diversity > 0

    def test_config_override(self):
        """Test configuration override."""
        config = {
            "similarity_type": "ldp",
            "split_sentences": True,
        }
        metric = DependencyParse(config)

        assert metric.config.similarity_type == "ldp"
        assert metric.config.split_sentences is True

    @pytest.mark.slow
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        metric = DependencyParse()
        diversity = metric([])
        assert diversity == 0.0


class TestConstituencyParse:
    """Tests for ConstituencyParse metric."""

    @requires_benepar
    @requires_karateclub
    @pytest.mark.slow
    def test_basic_diversity(self, sample_corpus):
        """Test basic constituency parse diversity."""
        metric = ConstituencyParse({"similarity_type": "ldp"})
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity > 0
        assert diversity < 10

    @requires_benepar
    @requires_karateclub
    @pytest.mark.slow
    def test_similarity(self, sample_corpus):
        """Test similarity calculation."""
        metric = ConstituencyParse({"similarity_type": "feather"})
        similarity = metric.similarity(sample_corpus)

        # Similarity should be between 0 and 1
        assert 0 <= similarity <= 1


class TestConstituencyTreeConstruction:
    """Regression tests for constituency parse tree construction.

    These cover a bug where ``hasattr(span, "_.parse_string")`` was used to detect
    benepar's spaCy extension. That check can never be true -- the attribute is
    named ``_``, not ``_.parse_string`` -- so every sentence silently fell back to a
    single-node graph. The metric then returned exactly 1.0 for any input under
    ``ldp``, and raised KeyError under ``tree_edit_distance``.
    """

    @requires_benepar
    @pytest.mark.slow
    def test_tree_has_real_structure(self):
        """A multi-word sentence must yield more than the fallback single node."""
        metric = ConstituencyParse({"verbose": False})
        graph = metric._generate_constituency_tree("the old man walked home slowly")

        assert graph.number_of_nodes() > 1, "fell back to the single-node graph"
        assert graph.number_of_edges() > 0
        labels = {d["label"] for _, d in graph.nodes(data=True)}
        # A real constituency parse contains phrase labels, not just a bare root
        assert {"NP", "VP"} & labels, f"no phrase labels found in {labels}"

    @requires_benepar
    @pytest.mark.slow
    def test_trees_differ_across_sentences(self):
        """Structurally different sentences must not produce identical trees."""
        metric = ConstituencyParse({"verbose": False})
        short = metric._generate_constituency_tree("dogs bark")
        long = metric._generate_constituency_tree(
            "although it rained, she went outside because she wanted air"
        )
        assert short.number_of_nodes() != long.number_of_nodes()

    @requires_benepar
    @pytest.mark.slow
    def test_discriminates_between_corpora(self):
        """Syntactically varied text must score above syntactically uniform text."""
        metric = ConstituencyParse({"verbose": False})
        uniform = [
            "a violent tempest wrecked our village",
            "the fierce gale devastated their settlement",
            "that savage hurricane destroyed this community",
        ]
        varied = [
            "dogs bark",
            "the extremely old man who lived by the sea walked home",
            "although it rained, she went outside because she wanted air",
        ]
        assert metric(varied) > metric(uniform)


class TestTreeEditDistanceHelpers:
    """Tests for the ZSS conversion helper."""

    def test_edgeless_tree_yields_root_node(self):
        """A lone root must still be convertible, or callers raise KeyError."""
        import networkx as nx

        from linguistic_diversity.diversities.syntactic import _get_tree_nodes_dict

        graph = nx.DiGraph()
        graph.add_node(0)
        nodes = _get_tree_nodes_dict(graph)

        assert 0 in nodes, "edgeless tree produced no node entry"
