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
        graph.add_node(0, pos="NOUN")
        nodes = _get_tree_nodes_dict(graph, graph)

        assert 0 in nodes, "edgeless tree produced no node entry"
        assert nodes[0].label == "NOUN:ROOT", "root node lost its part-of-speech label"

    def test_labels_carry_pos_and_dependency_not_token_index(self):
        """Node labels must describe structure, not position.

        Labelling ZSS nodes by token index made the edit distance blind to part of
        speech and grammatical function: any two parses sharing a shape compared
        as identical, so an intransitive with an adverbial matched a transitive
        with a direct object.
        """
        from linguistic_diversity import DependencyParse
        from linguistic_diversity.diversities.syntactic import _tree_edit_distance

        metric = DependencyParse({"similarity_type": "tree_edit_distance"})
        intransitive = metric._generate_dependency_tree("She sings beautifully.")
        transitive = metric._generate_dependency_tree("Dogs eat bones.")

        assert (
            _tree_edit_distance(intransitive, transitive) > 0
        ), "same tree shape with different POS and dependencies compared as identical"

    def test_same_frame_different_words_still_collapses(self):
        """The converse: identical structure must stay identical across lexicalisations."""
        from linguistic_diversity import DependencyParse
        from linguistic_diversity.diversities.syntactic import _tree_edit_distance

        metric = DependencyParse({"similarity_type": "tree_edit_distance"})
        first = metric._generate_dependency_tree("The tall boy kicked the ball.")
        second = metric._generate_dependency_tree("A red car struck the fence.")

        assert (
            _tree_edit_distance(first, second) == 0
        ), "one syntactic frame with different words no longer compares as identical"


class TestScaleFreeSimilarity:
    """Parse similarity must not depend on sentence length.

    exp(-edit_distance) is not scale-free. Edit distance grows with sentence
    length, so on ordinary text every off-diagonal entry underflowed to ~0, Z
    became the identity, and diversity saturated at the document count. The
    metric was counting sentences, not comparing structures -- and it looked
    healthy on the validation benchmark only because sentences sharing a frame
    there have distance exactly 0.
    """

    @staticmethod
    def _realistic_corpus():
        return [
            "escape plan we haven't thought of yet.",
            "omelet that is the most amazing ever.",
            "airplane ticket that's even cheaper.",
            "actual deadline for this paper.",
            "event that we can go to this weekend.",
        ]

    def test_does_not_saturate_on_ordinary_sentences(self):
        """Pinned to the Hill index: the threshold is calibrated to that quantity.

        The defect this guards against lives in the similarity matrix, not the
        index -- a Z collapsed to the identity. Vendi reads the same matrix higher
        (4.37 against 2.63 here) because it discounts similarity less
        aggressively, so a fraction-of-ceiling bound written for one index is
        meaningless for the other. The index-independent version of this check is
        test_off_diagonal_similarity_is_not_underflowed below.
        """
        from linguistic_diversity import DependencyParse

        corpus = self._realistic_corpus()
        diversity = DependencyParse({"index": "hill", "verbose": False})(corpus)

        assert diversity < len(corpus) * 0.6, (
            f"diversity {diversity:.3f} is close to the document count "
            f"{len(corpus)}: the similarity matrix has collapsed to the identity"
        )

    def test_off_diagonal_similarity_is_not_underflowed(self):
        import numpy as np

        from linguistic_diversity import DependencyParse

        metric = DependencyParse({"verbose": False})
        features, _ = metric.extract_features(self._realistic_corpus())
        Z = metric.calculate_similarities(features)
        off_diagonal = Z[~np.eye(Z.shape[0], dtype=bool)]

        assert off_diagonal.mean() > 0.05, (
            f"mean off-diagonal similarity {off_diagonal.mean():.2e} -- these are all "
            "short English sentences and should not be mutually unrecognisable"
        )
        assert Z.min() >= 0.0 and Z.max() <= 1.0
        assert np.allclose(np.diag(Z), 1.0)

    def test_similarity_is_invariant_to_sentence_length(self):
        """The same structural contrast must score the same at any length.

        Two sentences sharing a frame, and two differing in the same way, should
        compare alike whether they are short or long. Under exp(-distance) the
        long pair scored far lower purely because more tokens meant more edits.
        """
        from linguistic_diversity import DependencyParse
        from linguistic_diversity.diversities.syntactic import _tree_edit_similarity

        metric = DependencyParse({"verbose": False})
        short = ["The boy kicked the ball.", "A car struck the fence."]
        long = [
            "The determined boy from the village kicked the muddy ball over the fence.",
            "A speeding car from the highway struck the wooden fence beside the road.",
        ]
        short_sim = _tree_edit_similarity(*[metric._generate_dependency_tree(s) for s in short])
        long_sim = _tree_edit_similarity(*[metric._generate_dependency_tree(s) for s in long])

        assert abs(short_sim - long_sim) < 0.35, (
            f"same structural contrast scored {short_sim:.3f} when short and "
            f"{long_sim:.3f} when long: similarity still depends on length"
        )
