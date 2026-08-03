"""Syntactic diversity metrics based on parse tree structures.

This module provides metrics for measuring diversity in the syntactic structure of text
using dependency and constituency parse trees.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, cast

import faiss
import networkx as nx
import numpy as np
import numpy.typing as npt
import spacy
import zss
from sklearn.decomposition import PCA
from spacy.tokens import Span

# Karateclub is optional - only needed for ldp/feather similarity types
try:
    from karateclub import LDP, FeatherGraph

    KARATECLUB_AVAILABLE = True
except ImportError:
    KARATECLUB_AVAILABLE = False
    FeatherGraph = None
    LDP = None

from ..metric import MetricConfig, TextDiversity
from ..utils import (
    clean_text,
    compute_similarity_matrix_faiss,
    compute_similarity_matrix_pairwise,
    split_sentences,
)


@dataclass
class SyntacticConfig(MetricConfig):
    """Configuration for syntactic diversity metrics."""

    # Similarity computation.
    # Default is tree_edit_distance: it needs no optional dependency and actually
    # discriminates. The "ldp"/"feather" graph embeddings require karateclub (which
    # pins numpy<1.23) and, because their histograms are compared by raw cosine,
    # return ~1.0 even for wildly different parses -- see benchmarks/ for measurements.
    similarity_type: str = "tree_edit_distance"
    n_components: int | str | None = None  # PCA dimensions ("auto" or int)

    # Sentence processing
    split_sentences: bool = False


# Model caching
_SPACY_MODEL_CACHE: dict[str, Any] = {}


@lru_cache(maxsize=1)
def _get_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Get or load a cached spaCy model.

    Args:
        model_name: Name of the spaCy model.

    Returns:
        Loaded spaCy model.
    """
    if model_name not in _SPACY_MODEL_CACHE:
        _SPACY_MODEL_CACHE[model_name] = spacy.load(model_name)
    return _SPACY_MODEL_CACHE[model_name]


def _node_match_on_pos(node1: dict[str, Any], node2: dict[str, Any]) -> bool:
    """Match graph nodes based on POS tags."""
    return node1.get("pos") == node2.get("pos")


def _edge_match_on_dep(edge1: dict[str, Any], edge2: dict[str, Any]) -> bool:
    """Match graph edges based on dependency relations."""
    return edge1.get("dep") == edge2.get("dep")


def _sort_key(node: Any) -> tuple[int, Any]:
    """Order sibling nodes by token index, so trees are compared in surface order."""
    try:
        return (0, int(node))
    except (TypeError, ValueError):
        return (1, str(node))


def _zss_label(source: nx.DiGraph, node: Any, parent: Any | None) -> str:
    """Label a node by its part of speech and its relation to its head.

    zss compares nodes by label, so this is what decides whether two parses count
    as the same structure. Token identity is deliberately excluded -- two
    sentences sharing a structure should match regardless of their words.
    """
    pos = source.nodes.get(node, {}).get("pos", "X")
    if parent is None:
        return f"{pos}:ROOT"
    dep = source.edges.get((parent, node), {}).get("dep", "dep")
    return f"{pos}:{dep}"


def _get_tree_nodes_dict(tree: nx.DiGraph, source: nx.DiGraph) -> dict[Any, zss.Node]:
    """Build ZSS node dictionary from tree edges.

    Args:
        tree: Directed graph representing a tree (attributes not preserved by
            ``nx.dfs_tree``, hence ``source``).
        source: The original parse graph, carrying ``pos`` and ``dep`` attributes.

    Returns:
        Dictionary mapping node IDs to ZSS nodes.
    """
    parents = {child: parent for parent, child in tree.edges()}
    # Every node gets a node object first, so a lone root (no edges) still resolves
    # and the caller's nodes_dict[root] lookup cannot raise KeyError.
    nodes_dict: dict[Any, zss.Node] = {
        node: zss.Node(_zss_label(source, node, parents.get(node))) for node in tree.nodes()
    }
    # zss.simple_distance is an *ordered* tree edit distance, so sibling order is
    # part of the comparison. Sort by token index to make it surface order rather
    # than whatever order the graph happens to iterate in.
    for parent, child in sorted(tree.edges(), key=lambda e: (_sort_key(e[0]), _sort_key(e[1]))):
        nodes_dict[parent].addkid(nodes_dict[child])
    return nodes_dict


def _normalized_edit_similarity(graph1: nx.DiGraph, graph2: nx.DiGraph, distance: float) -> float:
    """Turn an edit distance into a similarity in [0, 1], free of tree size.

    The bound is ``max(|T1|, |T2|)``, not the sum: relabelling is a single edit,
    so the worst case is relabelling every node of the smaller tree and inserting
    the size difference. Dividing by the sum instead would floor two
    maximally-different equal-sized trees at 0.5 and compress every score into the
    upper half of the range.

    The previous conversion, ``exp(-distance)``, was not scale-free, and that
    made the metric useless on realistic text. Edit distance grows with sentence
    length: on the 6-9 token responses in McDiv, distances of 4-8 are ordinary,
    and ``exp(-6)`` is 0.002. Every off-diagonal entry underflowed to ~0, Z
    became the identity, and diversity saturated at the document count -- the
    metric was counting sentences rather than comparing structures. It looked
    healthy on benchmarks/metric_validation/ only because sentences sharing a
    frame there have distance exactly 0, and those zeros carried all the signal.
    """
    denom = max(graph1.number_of_nodes(), graph2.number_of_nodes())
    if denom == 0:
        return 1.0
    return float(min(max(1.0 - distance / denom, 0.0), 1.0))


def _tree_edit_similarity(graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
    """Size-normalised tree edit similarity in [0, 1]."""
    return _normalized_edit_similarity(graph1, graph2, _tree_edit_distance(graph1, graph2))


def _graph_edit_similarity_fn(**kwargs: Any) -> Callable[[nx.DiGraph, nx.DiGraph], float]:
    """Size-normalised wrapper around networkx graph edit distance."""
    raw = partial(nx.graph_edit_distance, **kwargs)

    def similarity(graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        return _normalized_edit_similarity(graph1, graph2, float(raw(graph1, graph2)))

    return similarity


def _tree_edit_distance(graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
    """Compute tree edit distance using ZSS algorithm.

    Args:
        graph1: First dependency graph.
        graph2: Second dependency graph.

    Returns:
        Tree edit distance.
    """
    # Find root nodes (nodes with no incoming edges)
    root1 = [n for n, d in graph1.in_degree() if d == 0][0]
    root2 = [n for n, d in graph2.in_degree() if d == 0][0]

    # Convert to DFS trees
    tree1 = nx.dfs_tree(graph1, source=root1)
    tree2 = nx.dfs_tree(graph2, source=root2)

    # Build ZSS node dictionaries. The dfs_tree copies carry no attributes, so the
    # original graphs are passed alongside to supply node labels.
    nodes1 = _get_tree_nodes_dict(tree1, graph1)
    nodes2 = _get_tree_nodes_dict(tree2, graph2)

    # Compute edit distance
    return float(zss.simple_distance(nodes1[root1], nodes2[root2]))


class DependencyParse(TextDiversity["npt.NDArray[Any] | list[nx.DiGraph]"]):
    """Dependency parse tree diversity.

    This metric computes diversity based on the structure of dependency parse trees.
    Multiple similarity computation methods are supported:
    - "ldp": Local Degree Profile (fast, scalable)
    - "feather": FeatherGraph embedding (fast, scalable)
    - "tree_edit_distance": Zhang-Shasha edit distance (slow, exact)
    - "graph_edit_distance": Graph edit distance (very slow, exact)

    Example:
        >>> metric = DependencyParse({"similarity_type": "ldp"})
        >>> corpus = ['The cat sat', 'A dog ran', 'Birds fly']
        >>> diversity = metric(corpus)
    """

    # Narrow the base annotation so attribute access type-checks
    config: SyntacticConfig

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize dependency parse diversity metric.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)
        self.model = _get_spacy_model()

    @classmethod
    def _config_class(cls) -> type[SyntacticConfig]:
        return SyntacticConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "similarity_type": "tree_edit_distance",
            "split_sentences": False,
        }

    def _generate_dependency_tree(self, text: str) -> nx.DiGraph:
        """Generate dependency parse tree for text.

        Args:
            text: Input text.

        Returns:
            Directed graph representing dependency tree.
        """
        doc = self.model(text)

        graph = nx.DiGraph()

        # Add nodes with attributes
        nodes = [(str(token.i), {"text": token.text, "pos": token.pos_}) for token in doc]
        graph.add_nodes_from(nodes)

        # Add edges with dependency labels
        edges = [
            (str(token.head.i), str(token.i), {"dep": token.dep_})
            for token in doc
            if token.head.i != token.i
        ]
        graph.add_edges_from(edges)

        return graph

    def extract_features(
        self, corpus: list[str]
    ) -> tuple[npt.NDArray[Any] | list[nx.DiGraph], list[str]]:
        """Extract dependency parse trees from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (features, documents). Features are either graph embeddings
            (numpy array) or list of graphs depending on similarity_type.
        """
        # Clean corpus
        corpus = clean_text(corpus)

        # Optionally split into sentences
        if self.config.split_sentences:
            corpus = split_sentences(corpus)

        # Generate dependency trees
        graphs = [self._generate_dependency_tree(text) for text in corpus]

        # For graph/tree edit distance, return graphs directly
        if "distance" in self.config.similarity_type:
            return graphs, corpus

        # For embedding methods, convert to embeddings
        # Convert node labels to integers (required by karateclub)
        graphs_int = [nx.convert_node_labels_to_integers(g, first_label=0) for g in graphs]

        # Compute graph embeddings
        if self.config.similarity_type == "ldp":
            if not KARATECLUB_AVAILABLE:
                raise ImportError(
                    "karateclub is required for 'ldp' similarity. "
                    "Install it with: pip install karateclub\n"
                    "Note: karateclub has older dependencies. Use 'tree_edit_distance' instead."
                )
            model = LDP(bins=64)
            model.fit(graphs_int)
            embeddings = model.get_embedding().astype(np.float32)
        elif self.config.similarity_type == "feather":
            if not KARATECLUB_AVAILABLE:
                raise ImportError(
                    "karateclub is required for 'feather' similarity. "
                    "Install it with: pip install karateclub\n"
                    "Note: karateclub has older dependencies. Use 'tree_edit_distance' instead."
                )
            model = FeatherGraph(theta_max=100)
            model.fit(graphs_int)
            embeddings = model.get_embedding().astype(np.float32)
        else:
            raise ValueError(
                f"Unknown similarity_type: {self.config.similarity_type}. "
                f"Use 'ldp', 'feather', 'tree_edit_distance', or 'graph_edit_distance'."
            )

        # Optional PCA dimensionality reduction
        if self.config.n_components is not None and len(embeddings) > 1:
            if self.config.n_components == "auto":
                n_comp = min(max(2, len(embeddings) // 10), embeddings.shape[-1])
                if self.config.verbose:
                    print(f"Using n_components={n_comp}")
            else:
                n_comp = int(self.config.n_components)

            if 0 < n_comp < embeddings.shape[-1]:
                embeddings = PCA(n_components=n_comp).fit_transform(embeddings)

        return embeddings, corpus

    def calculate_similarities(
        self, features: npt.NDArray[np.float64] | list[nx.DiGraph]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities between parse trees.

        Args:
            features: Either embeddings (numpy array) or graphs (list).

        Returns:
            Similarity matrix (n x n).
        """
        # For edit distance methods, compute pairwise distances
        if "distance" in self.config.similarity_type:
            # Warn about computational complexity for large corpora
            if self.config.similarity_type == "graph_edit_distance" and len(features) > 10:
                raise ValueError(
                    f"graph_edit_distance has O(n!) complexity and is impractical for "
                    f"{len(features)} documents (max recommended: 10). "
                    f"Use 'ldp', 'feather', or 'tree_edit_distance' instead."
                )

            sim_fn: Callable[[nx.DiGraph, nx.DiGraph], float]
            if self.config.similarity_type == "tree_edit_distance":
                sim_fn = _tree_edit_similarity
            elif self.config.similarity_type == "graph_edit_distance":
                sim_fn = _graph_edit_similarity_fn(
                    node_match=_node_match_on_pos,
                    edge_match=_edge_match_on_dep,
                )
            else:
                raise ValueError(f"Unknown distance type: {self.config.similarity_type}")

            # sim_fn already returns a size-normalised similarity in [0, 1], so the
            # diagonal is 1.0 and no exponential decay is applied afterwards.
            graphs = cast("list[nx.DiGraph]", features)
            Z = compute_similarity_matrix_pairwise(
                graphs,
                sim_fn,
                diagonal_val=1.0,
                verbose=self.config.verbose,
            )

        # For embedding methods, use FAISS
        else:
            embeddings = cast("npt.NDArray[np.float64]", features)
            Z = compute_similarity_matrix_faiss(
                embeddings,
                distance_metric=faiss.METRIC_INNER_PRODUCT,
                postprocess=None,
            )

        return Z

    def calculate_abundance(self, species: list[str]) -> npt.NDArray[np.float64]:
        """Calculate uniform abundance distribution.

        Args:
            species: List of documents.

        Returns:
            Uniform distribution over documents.
        """
        n = len(species)
        return np.full(n, 1.0 / n, dtype=np.float64)


class ConstituencyParse(TextDiversity["npt.NDArray[Any] | list[nx.DiGraph]"]):
    """Constituency parse tree diversity.

    This metric computes diversity based on constituency (phrase structure) parse trees.
    Requires benepar to be installed.

    Example:
        >>> metric = ConstituencyParse()
        >>> corpus = ['The cat sat', 'A dog ran']
        >>> diversity = metric(corpus)
    """

    # Narrow the base annotation so attribute access type-checks
    config: SyntacticConfig

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize constituency parse diversity metric.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)
        self.model = _get_spacy_model()

        # Add benepar to pipeline
        if "benepar" not in self.model.pipe_names:
            import benepar

            # Download model if needed
            try:
                self.model.add_pipe("benepar", config={"model": "benepar_en3"})
            except Exception:
                # Try downloading first
                benepar.download("benepar_en3")
                self.model.add_pipe("benepar", config={"model": "benepar_en3"})

    @classmethod
    def _config_class(cls) -> type[SyntacticConfig]:
        return SyntacticConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "similarity_type": "tree_edit_distance",
            "split_sentences": False,
        }

    def _generate_constituency_tree(self, text: str) -> nx.DiGraph:
        """Generate constituency parse tree for text.

        Args:
            text: Input text.

        Returns:
            Directed graph representing constituency tree.
        """
        doc = self.model(text)

        # Get first sentence
        sent = list(doc.sents)[0] if list(doc.sents) else doc

        # Convert parse tree to networkx graph
        graph = nx.DiGraph()

        # benepar exposes the parse through spaCy extensions on Span: ``labels``
        # gives a node's constituent labels and ``children`` its sub-constituents.
        # Note hasattr(span, "_.labels") is always False -- the attribute is named
        # "_", not "_.labels" -- so extension presence must be checked on the class.
        if not Span.has_extension("labels"):
            raise RuntimeError(
                "benepar did not register its spaCy extensions, so no constituency "
                "parse is available. Install it with: pip install benepar"
            )

        counter = itertools.count()

        def add_span(span: Any, parent_id: int | None = None) -> None:
            """Recursively add a constituent and its children to the graph."""
            node_id = next(counter)

            labels = tuple(span._.labels)
            if labels:
                label = labels[0]
            elif len(span) == 1:
                label = span[0].tag_  # leaf: use the POS tag
            else:
                label = "X"

            graph.add_node(node_id, label=label)
            if parent_id is not None:
                graph.add_edge(parent_id, node_id)

            for child in span._.children:
                add_span(child, node_id)

        add_span(sent)

        return graph

    def extract_features(
        self, corpus: list[str]
    ) -> tuple[npt.NDArray[Any] | list[nx.DiGraph], list[str]]:
        """Extract constituency parse trees from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (features, documents).
        """
        # Clean corpus
        corpus = clean_text(corpus)

        # Optionally split into sentences
        if self.config.split_sentences:
            corpus = split_sentences(corpus)

        # Generate constituency trees
        graphs = [self._generate_constituency_tree(text) for text in corpus]

        # For edit distance, return graphs
        if "distance" in self.config.similarity_type:
            return graphs, corpus

        # For embeddings, convert to integer labels and embed
        graphs_int = [nx.convert_node_labels_to_integers(g, first_label=0) for g in graphs]

        if self.config.similarity_type == "ldp":
            if not KARATECLUB_AVAILABLE:
                raise ImportError(
                    "karateclub is required for 'ldp' similarity. "
                    "Install it with: pip install karateclub\n"
                    "Note: karateclub has older dependencies. Use 'tree_edit_distance' instead."
                )
            model = LDP(bins=64)
            model.fit(graphs_int)
            embeddings = model.get_embedding().astype(np.float32)
        elif self.config.similarity_type == "feather":
            if not KARATECLUB_AVAILABLE:
                raise ImportError(
                    "karateclub is required for 'feather' similarity. "
                    "Install it with: pip install karateclub\n"
                    "Note: karateclub has older dependencies. Use 'tree_edit_distance' instead."
                )
            model = FeatherGraph(theta_max=100)
            model.fit(graphs_int)
            embeddings = model.get_embedding().astype(np.float32)
        else:
            raise ValueError(f"Unknown similarity_type: {self.config.similarity_type}")

        # Optional PCA
        if self.config.n_components is not None and len(embeddings) > 1:
            if self.config.n_components == "auto":
                n_comp = min(max(2, len(embeddings) // 10), embeddings.shape[-1])
            else:
                n_comp = int(self.config.n_components)

            if 0 < n_comp < embeddings.shape[-1]:
                embeddings = PCA(n_components=n_comp).fit_transform(embeddings)

        return embeddings, corpus

    def calculate_similarities(
        self, features: npt.NDArray[np.float64] | list[nx.DiGraph]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities between parse trees.

        Args:
            features: Either embeddings or graphs.

        Returns:
            Similarity matrix (n x n).
        """
        if "distance" in self.config.similarity_type:
            # Warn about computational complexity
            if self.config.similarity_type == "graph_edit_distance" and len(features) > 10:
                raise ValueError(
                    f"graph_edit_distance has O(n!) complexity and is impractical for "
                    f"{len(features)} documents (max recommended: 10). "
                    f"Use 'ldp', 'feather', or 'tree_edit_distance' instead."
                )

            sim_fn: Callable[[nx.DiGraph, nx.DiGraph], float]
            if self.config.similarity_type == "tree_edit_distance":
                sim_fn = _tree_edit_similarity
            else:
                sim_fn = _graph_edit_similarity_fn()

            # sim_fn already returns a size-normalised similarity in [0, 1], so the
            # diagonal is 1.0 and no exponential decay is applied afterwards.
            graphs = cast("list[nx.DiGraph]", features)
            Z = compute_similarity_matrix_pairwise(
                graphs,
                sim_fn,
                diagonal_val=1.0,
                verbose=self.config.verbose,
            )
        else:
            embeddings = cast("npt.NDArray[np.float64]", features)
            Z = compute_similarity_matrix_faiss(
                embeddings,
                distance_metric=faiss.METRIC_INNER_PRODUCT,
                postprocess=None,
            )

        return Z

    def calculate_abundance(self, species: list[str]) -> npt.NDArray[np.float64]:
        """Calculate uniform abundance distribution.

        Args:
            species: List of documents.

        Returns:
            Uniform distribution over documents.
        """
        n = len(species)
        return np.full(n, 1.0 / n, dtype=np.float64)
