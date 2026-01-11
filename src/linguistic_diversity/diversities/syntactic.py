"""Syntactic diversity metrics based on parse tree structures.

This module provides metrics for measuring diversity in the syntactic structure of text
using dependency and constituency parse trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any

import faiss  # type: ignore
import networkx as nx
import numpy as np
import numpy.typing as npt
import spacy
import zss  # type: ignore
from karateclub import FeatherGraph, LDP  # type: ignore
from sklearn.decomposition import PCA

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

    # Similarity computation
    similarity_type: str = "ldp"  # "ldp", "feather", "tree_edit_distance", "graph_edit_distance"
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


def _get_tree_nodes_dict(tree: nx.DiGraph) -> dict[Any, zss.Node]:
    """Build ZSS node dictionary from tree edges.

    Args:
        tree: Directed graph representing a tree.

    Returns:
        Dictionary mapping node IDs to ZSS nodes.
    """
    nodes_dict: dict[Any, zss.Node] = {}
    for parent, child in tree.edges():
        if parent not in nodes_dict:
            nodes_dict[parent] = zss.Node(parent)
        if child not in nodes_dict:
            nodes_dict[child] = zss.Node(child)
        nodes_dict[parent].addkid(nodes_dict[child])
    return nodes_dict


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

    # Build ZSS node dictionaries
    nodes1 = _get_tree_nodes_dict(tree1)
    nodes2 = _get_tree_nodes_dict(tree2)

    # Compute edit distance
    return float(zss.simple_distance(nodes1[root1], nodes2[root2]))


class DependencyParse(TextDiversity):
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
            "similarity_type": "ldp",
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
        nodes = [
            (str(token.i), {"text": token.text, "pos": token.pos_})
            for token in doc
        ]
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
            return graphs, corpus  # type: ignore

        # For embedding methods, convert to embeddings
        # Convert node labels to integers (required by karateclub)
        graphs_int = [
            nx.convert_node_labels_to_integers(g, first_label=0)
            for g in graphs
        ]

        # Compute graph embeddings
        if self.config.similarity_type == "ldp":
            model = LDP(bins=64)
            model.fit(graphs_int)
            embeddings = model.get_embedding().astype(np.float32)
        elif self.config.similarity_type == "feather":
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
            if self.config.similarity_type == "tree_edit_distance":
                dist_fn = _tree_edit_distance
            elif self.config.similarity_type == "graph_edit_distance":
                dist_fn = partial(
                    nx.graph_edit_distance,
                    node_match=_node_match_on_pos,
                    edge_match=_edge_match_on_dep,
                )
            else:
                raise ValueError(f"Unknown distance type: {self.config.similarity_type}")

            # Compute distance matrix
            Z = compute_similarity_matrix_pairwise(
                features,  # type: ignore
                dist_fn,
                diagonal_val=0.0,  # Distance to self is 0
                verbose=self.config.verbose,
            )

            # Convert distances to similarities (exponential decay)
            Z = np.exp(-Z)

        # For embedding methods, use FAISS
        else:
            Z = compute_similarity_matrix_faiss(
                features,  # type: ignore
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


class ConstituencyParse(TextDiversity):
    """Constituency parse tree diversity.

    This metric computes diversity based on constituency (phrase structure) parse trees.
    Requires benepar to be installed.

    Example:
        >>> metric = ConstituencyParse()
        >>> corpus = ['The cat sat', 'A dog ran']
        >>> diversity = metric(corpus)
    """

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
            "similarity_type": "ldp",
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

        def add_tree_to_graph(node: Any, parent_id: int | None = None, node_id: int = 0) -> int:
            """Recursively add parse tree nodes to graph."""
            current_id = node_id

            if hasattr(node, "_.labels"):
                label = node._.labels[0] if node._.labels else "ROOT"
            else:
                label = node.label_() if hasattr(node, "label_") else str(node)

            graph.add_node(current_id, label=label)

            if parent_id is not None:
                graph.add_edge(parent_id, current_id)

            node_id += 1

            # Recursively add children
            if hasattr(node, "__iter__") and not isinstance(node, str):
                for child in node:
                    node_id = add_tree_to_graph(child, current_id, node_id)

            return node_id

        # Build graph from parse tree
        if hasattr(sent, "_.parse_string"):
            # Use benepar parse
            add_tree_to_graph(sent._.parse_tree)
        else:
            # Fallback: single node
            graph.add_node(0, label="S")

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
            return graphs, corpus  # type: ignore

        # For embeddings, convert to integer labels and embed
        graphs_int = [
            nx.convert_node_labels_to_integers(g, first_label=0)
            for g in graphs
        ]

        if self.config.similarity_type == "ldp":
            model = LDP(bins=64)
            model.fit(graphs_int)
            embeddings = model.get_embedding().astype(np.float32)
        elif self.config.similarity_type == "feather":
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
            if self.config.similarity_type == "tree_edit_distance":
                dist_fn = _tree_edit_distance
            else:
                dist_fn = partial(nx.graph_edit_distance)

            Z = compute_similarity_matrix_pairwise(
                features,  # type: ignore
                dist_fn,
                diagonal_val=0.0,
                verbose=self.config.verbose,
            )
            Z = np.exp(-Z)
        else:
            Z = compute_similarity_matrix_faiss(
                features,  # type: ignore
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
