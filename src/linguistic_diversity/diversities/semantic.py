"""Semantic diversity metrics based on distributional semantics.

This module provides metrics for measuring diversity in the semantic content of text
using contextualized and static word embeddings.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Any, cast

import faiss
import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from transformers import logging as transformers_logging

from ..metric import MetricConfig, TextDiversity
from ..utils import (
    chunker,
    compute_similarity_matrix_faiss,
    detect_subword_scheme,
    merge_bpe,
    similarity_search_faiss,
)

# Suppress transformers warnings
transformers_logging.set_verbosity_error()


@dataclass
class SemanticConfig(MetricConfig):
    """Configuration for semantic diversity metrics."""

    # Similarity computation
    distance_fn: int = faiss.METRIC_INNER_PRODUCT
    scale_dist: str | None = None
    power_reg: bool = False
    mean_adj: bool = True

    # Feature processing
    remove_stopwords: bool = False
    remove_punct: bool = False
    n_components: int | str | None = None  # PCA dimensions ("auto" or int)

    # Model settings
    model_name: str = "bert-base-uncased"
    batch_size: int = 16
    use_cuda: bool = True
    trust_remote_code: bool = False  # Required by models shipping custom code
    # Extra arguments forwarded to SentenceTransformer.encode. Task-conditioned and
    # instruction-tuned embedders require these, e.g. {"task": "text-matching"} for
    # jina-v3/v5 or {"prompt_name": "query"} for models with named prompts.
    encode_kwargs: dict[str, Any] = field(default_factory=dict)


# Model caching to avoid reloading
_MODEL_CACHE: dict[str, Any] = {}


def _get_cached_model(
    model_name: str,
    model_loader: Callable[..., Any],
    cache_scope: str = "",
    **kwargs: Any,
) -> Any:
    """Get or load a cached model.

    Args:
        model_name: Name/path of the model.
        model_loader: Callable that loads the model (a class or a from_pretrained).
        cache_scope: Extra cache-key component that is NOT passed to the loader.
            Used for the target device: callers move the returned module onto their
            device, which mutates the shared instance, so models bound for different
            devices must not share a cache entry.
        **kwargs: Extra loader arguments (also part of the cache key, so that the
            same checkpoint loaded with different options is not aliased).

    Returns:
        Loaded model.
    """
    opts = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    loader_name = getattr(model_loader, "__qualname__", repr(model_loader))
    cache_key = f"{loader_name}:{model_name}:{opts}:{cache_scope}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = model_loader(model_name, **kwargs)
    return _MODEL_CACHE[cache_key]


def clear_model_cache() -> None:
    """Release every cached model and free the GPU memory they held.

    Models are cached indefinitely so repeated metric construction is cheap. That
    is the wrong trade-off when sweeping many checkpoints in one process: the cache
    grows without bound and will exhaust host RAM or VRAM. Call this between models
    in a sweep.
    """
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_token_states(outputs: Any, expected_len: int) -> torch.Tensor | None:
    """Pull per-token hidden states out of whatever a model returned.

    Encoders differ in what they hand back: standard HF models expose
    ``hidden_states``/``last_hidden_state``, while embedding models that ship
    custom code often return a single pooled vector per document.

    Args:
        outputs: Raw model output.
        expected_len: Sequence length the states must have to be per-token.

    Returns:
        A (batch x seq_len x hidden) tensor, or None if the output is not
        per-token (e.g. an already-pooled document embedding).
    """
    hidden = getattr(outputs, "hidden_states", None)
    if hidden:
        # Second-to-last layer is often better for semantic tasks than the last
        candidate = hidden[-2] if len(hidden) >= 2 else hidden[-1]
        if candidate.ndim == 3 and candidate.shape[1] == expected_len:
            return cast("torch.Tensor", candidate)

    last = getattr(outputs, "last_hidden_state", None)
    if last is not None and last.ndim == 3 and last.shape[1] == expected_len:
        return cast("torch.Tensor", last)

    if torch.is_tensor(outputs) and outputs.ndim == 3 and outputs.shape[1] == expected_len:
        return outputs

    return None


@lru_cache(maxsize=1)
def _cuda_is_usable() -> bool:
    """Check that CUDA can actually allocate, not merely that a device is listed.

    torch.cuda.is_available() only reports that a driver and device were found. A
    stale driver, an exclusive compute mode, or a GPU already at capacity all still
    report available and then fail on the first allocation. Probing once here lets
    callers fall back to CPU instead of crashing partway through a corpus.

    Returns:
        True if a trivial CUDA allocation succeeds.
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
        warnings.warn(
            f"CUDA reports a device but allocation failed ({exc.__class__.__name__}: "
            f"{str(exc).splitlines()[0]}). Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


def _resolve_device(use_cuda: bool) -> torch.device:
    """Pick the device to run on, verifying CUDA before selecting it.

    Args:
        use_cuda: Whether the caller asked for GPU acceleration.

    Returns:
        The device to use.
    """
    return torch.device("cuda" if use_cuda and _cuda_is_usable() else "cpu")


@cache
def _get_stopwords() -> set[str]:
    """Get English stopwords (cached)."""
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError:
        # Download stopwords if not available
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))


class TokenSemantics(TextDiversity[npt.NDArray[np.float64]]):
    """Token-level semantic diversity using contextualized embeddings.

    This metric computes diversity based on contextualized token embeddings
    from transformer models like BERT. Each token occurrence is treated as
    a separate species.

    Example:
        >>> metric = TokenSemantics()
        >>> corpus = ['one massive earth', 'an enormous globe']
        >>> diversity = metric(corpus)
    """

    # Narrow the base annotation so attribute access type-checks
    config: SemanticConfig

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize token semantic diversity metric.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)

        # Resolve the device before loading, so it can scope the model cache
        self.device = _resolve_device(self.config.use_cuda)

        # Load model and tokenizer. trust_remote_code matters for checkpoints whose
        # config carries an auto_map: without it, transformers silently falls back to
        # the base architecture named by model_type and drops the custom layers.
        trust = self.config.trust_remote_code
        self.model = _get_cached_model(
            self.config.model_name,
            AutoModel.from_pretrained,
            cache_scope=str(self.device),
            trust_remote_code=trust,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=trust
        )

        # Special tokens to exclude. Decoder-style tokenizers (Qwen, GPT-2) have no
        # cls/sep and report None, which must not end up in the filter set.
        self.undesirable_tokens = {
            token_id
            for token_id in (
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
            )
            if token_id is not None
        }

        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
            self.model.eval()

        # Some embedding models return one pooled vector per document instead of
        # per-token states. Resolve a token-level path for those up front, so the
        # failure surfaces at construction rather than as a degenerate score.
        self._token_encoder = self._resolve_token_encoder()

    @classmethod
    def _config_class(cls) -> type[SemanticConfig]:
        return SemanticConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        # Cosine on L2-normalized embeddings, squared, then mean-adjusted. Squaring
        # sharpens the similarity contrast without distorting the ordering, which
        # roughly doubles the separation between a paraphrase corpus and a genuinely
        # varied one relative to the Chebyshev/exp alternative, at no measurable cost
        # to rank agreement with ground truth. See use_cases/embedder_selection/.
        return {
            "model_name": "bert-base-uncased",
            "batch_size": 16,
            "use_cuda": True,
            "distance_fn": faiss.METRIC_INNER_PRODUCT,
            "scale_dist": None,
            "mean_adj": True,
            "power_reg": True,
        }

    @torch.no_grad()
    def _resolve_token_encoder(self) -> Any:
        """Find a callable returning per-token states, probing the model to verify.

        Standard encoders need nothing here. Pooling encoders (embedding models that
        return one vector per document) are handled by calling their inner backbone
        directly, replaying any projection applied to the input embeddings first so
        the token states match what the full model would have pooled over.

        Returns:
            None if the model already yields per-token states, otherwise a callable
            taking (input_ids, attention_mask) and returning them.

        Raises:
            ValueError: If the model pools its output and no token-level path
                can be found, since any score computed from it would be
                meaningless rather than merely imprecise.
        """
        if not isinstance(self.model, torch.nn.Module):
            return None

        probe = self.tokenizer(["probe text"], return_tensors="pt", padding=True, truncation=True)
        input_ids = probe.input_ids.to(self.device)
        attention_mask = probe.attention_mask.to(self.device)
        seq_len = input_ids.shape[1]

        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if _extract_token_states(outputs, seq_len) is not None:
            return None

        # Pooled output. Look for the inner transformer that ran before pooling.
        for attr in ("model", "encoder", "transformer", "bert", "backbone"):
            backbone = getattr(self.model, attr, None)
            if not isinstance(backbone, torch.nn.Module):
                continue
            embed_tokens = getattr(backbone, "embed_tokens", None)
            if embed_tokens is None:
                continue

            # Projections applied to the embeddings before the backbone (e.g. the
            # jasper_mlp in Jasper-style encoders) are part of the encoder and must
            # be replayed, or the token states come from a different space.
            projections = [
                module
                for name, module in self.model.named_children()
                if name.endswith("_mlp") and isinstance(module, torch.nn.Module)
            ]

            # Bind the loop variables as defaults so the returned closure keeps the
            # backbone it was built from rather than whatever the loop ends on.
            def encode(
                ids: torch.Tensor,
                mask: torch.Tensor,
                _embed: Any = embed_tokens,
                _projections: list[Any] = projections,
                _backbone: Any = backbone,
            ) -> torch.Tensor:
                embeds = _embed(ids)
                for projection in _projections:
                    embeds = projection(embeds)
                out = _backbone(inputs_embeds=embeds, attention_mask=mask)["last_hidden_state"]
                return cast("torch.Tensor", out)

            states = encode(input_ids, attention_mask)
            if states.ndim == 3 and states.shape[1] == seq_len:
                return encode

        raise ValueError(
            f"{self.config.model_name!r} returns pooled document embeddings "
            f"(shape {tuple(outputs.shape) if torch.is_tensor(outputs) else type(outputs).__name__}) "
            f"and no per-token path could be resolved, so token-level diversity "
            f"cannot be computed from it. Use DocumentSemantics for this model, or "
            f"pick an encoder that exposes hidden_states."
        )

    @torch.no_grad()
    def _encode_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of inputs.

        Args:
            input_ids: Token IDs (batch_size x seq_len).
            attention_mask: Attention mask (batch_size x seq_len).

        Returns:
            Contextualized embeddings (batch_size x seq_len x hidden_dim).
        """
        if self._token_encoder is not None:
            return cast("torch.Tensor", self._token_encoder(input_ids, attention_mask))

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        states = _extract_token_states(outputs, input_ids.shape[1])
        if states is None:
            raise ValueError(f"Could not extract per-token states from {self.config.model_name!r}.")
        return states

    def extract_features(self, corpus: list[str]) -> tuple[npt.NDArray[np.float64], list[str]]:
        """Extract token embeddings from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (embeddings, tokens).
        """
        # Tokenize all texts
        inputs = self.tokenizer(
            corpus,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Process in batches
        batches = zip(
            chunker(inputs.input_ids, self.config.batch_size),
            chunker(inputs.attention_mask, self.config.batch_size),
            strict=True,
        )

        embeddings_list = []
        for input_ids, attention_mask in batches:
            emb = self._encode_batch(
                input_ids.to(self.device),
                attention_mask.to(self.device),
            )
            embeddings_list.append(emb.cpu())

        # Combine batches
        all_embeddings = torch.cat(embeddings_list)

        # Flatten to (total_tokens x hidden_dim), tracking which document each
        # token came from so subword merging cannot run across documents
        n_docs, seq_len = inputs.input_ids.shape
        flat_ids = inputs.input_ids.view(-1).numpy()
        flat_embeddings = all_embeddings.view(-1, all_embeddings.shape[-1]).numpy()
        doc_ids = np.repeat(np.arange(n_docs), seq_len)

        # Filter out special tokens
        valid_mask = ~np.isin(flat_ids, list(self.undesirable_tokens))
        tokens_array = np.array(self.tokenizer.convert_ids_to_tokens(flat_ids))[valid_mask]
        embeddings_array = flat_embeddings[valid_mask]
        doc_ids = doc_ids[valid_mask]

        # Filter stopwords if requested
        if self.config.remove_stopwords:
            stopwords = _get_stopwords()
            keep_mask = ~np.isin(tokens_array, list(stopwords))
            tokens_array = tokens_array[keep_mask]
            embeddings_array = embeddings_array[keep_mask]
            doc_ids = doc_ids[keep_mask]

        # Filter punctuation if requested
        if self.config.remove_punct:
            punct_chars = set("""!()-[]{};:'",<>./?@#$%^&*_~""")
            keep_mask = ~np.isin(tokens_array, list(punct_chars))
            tokens_array = tokens_array[keep_mask]
            embeddings_array = embeddings_array[keep_mask]
            doc_ids = doc_ids[keep_mask]

        # Merge subwords so one word is one species, whichever convention the
        # tokenizer uses ("##" continuations vs. marked word starts)
        scheme = detect_subword_scheme(tokens_array)
        if scheme is not None:
            tokens_array, embeddings_array = merge_bpe(
                tokens_array, embeddings_array, scheme=scheme, group_ids=doc_ids
            )

        # Optional PCA dimensionality reduction
        if self.config.n_components is not None and len(embeddings_array) > 1:
            from sklearn.decomposition import PCA

            if self.config.n_components == "auto":
                n_comp = min(max(2, len(embeddings_array) // 10), embeddings_array.shape[-1])
            else:
                n_comp = int(self.config.n_components)

            if n_comp > 0 and n_comp < embeddings_array.shape[-1]:
                embeddings_array = PCA(n_components=n_comp).fit_transform(embeddings_array)

        return embeddings_array, tokens_array.tolist()

    def calculate_similarities(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities using FAISS.

        Args:
            features: Token embeddings (n_tokens x dim).

        Returns:
            Similarity matrix (n_tokens x n_tokens).
        """
        Z = compute_similarity_matrix_faiss(
            features,
            distance_metric=self.config.distance_fn,
            postprocess=self.config.scale_dist,
        )

        # Apply power regularization if requested
        if self.config.power_reg:
            Z = np.power(Z, 2)

        # Apply mean adjustment if requested
        if self.config.mean_adj:
            off_diag_mask = ~np.eye(Z.shape[0], dtype=bool)
            mean_sim = Z[off_diag_mask].mean()
            Z[off_diag_mask] -= mean_sim
            Z = np.maximum(Z, 0)  # Clip negative values

        return Z

    def calculate_abundance(self, species: list[str]) -> npt.NDArray[np.float64]:
        """Calculate uniform abundance distribution.

        Args:
            species: List of species (tokens).

        Returns:
            Uniform distribution over species.
        """
        n = len(species)
        return np.full(n, 1.0 / n, dtype=np.float64)


class DocumentSemantics(TextDiversity[npt.NDArray[np.float64]]):
    """Document-level semantic diversity using sentence embeddings.

    This metric computes diversity based on document-level embeddings
    from sentence transformer models, treating each document as a species.

    Example:
        >>> metric = DocumentSemantics()
        >>> corpus = ['one massive earth', 'an enormous globe']
        >>> diversity = metric(corpus)
    """

    # Narrow the base annotation so attribute access type-checks
    config: SemanticConfig

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize document semantic diversity metric.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)

        # Device setup
        self.device = _resolve_device(self.config.use_cuda)

        # Load sentence transformer model with caching. The device belongs in the
        # cache key: a SentenceTransformer is placed on its device at construction
        # and is not moved on retrieval, so omitting it would silently hand a
        # CUDA-resident model to a caller that asked for CPU.
        trust = self.config.trust_remote_code
        cache_key = (
            f"SentenceTransformer:{self.config.model_name}" f":trust={trust}:device={self.device}"
        )
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = SentenceTransformer(
                self.config.model_name,
                device=str(self.device),
                trust_remote_code=trust,
            )
        self.model = _MODEL_CACHE[cache_key]

    @classmethod
    def _config_class(cls) -> type[SemanticConfig]:
        return SemanticConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "model_name": "all-mpnet-base-v2",
            "batch_size": 32,
            "use_cuda": True,
            "distance_fn": faiss.METRIC_INNER_PRODUCT,
            "scale_dist": None,
            "mean_adj": False,
        }

    def extract_features(self, corpus: list[str]) -> tuple[npt.NDArray[np.float64], list[str]]:
        """Extract document embeddings from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (embeddings, documents).
        """
        # Encode documents. encode_kwargs carries whatever a task-conditioned or
        # instruction-tuned model requires (e.g. task="text-matching", prompt_name).
        embeddings = self.model.encode(
            corpus,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.verbose,
            convert_to_numpy=True,
            **self.config.encode_kwargs,
        )

        return embeddings, corpus

    def calculate_similarities(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate pairwise document similarities.

        Args:
            features: Document embeddings (n_docs x dim).

        Returns:
            Similarity matrix (n_docs x n_docs).
        """
        Z = compute_similarity_matrix_faiss(
            features,
            distance_metric=self.config.distance_fn,
            postprocess=self.config.scale_dist,
        )

        if self.config.power_reg:
            Z = np.power(Z, 2)

        if self.config.mean_adj:
            off_diag_mask = ~np.eye(Z.shape[0], dtype=bool)
            mean_sim = Z[off_diag_mask].mean()
            Z[off_diag_mask] -= mean_sim
            Z = np.maximum(Z, 0)

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

    def calculate_similarity_vector(
        self,
        query_features: npt.NDArray[np.float64],
        corpus_features: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate similarity between query and corpus documents.

        Args:
            query_features: Query document embedding (dim,).
            corpus_features: Corpus document embeddings (n_docs x dim).

        Returns:
            Similarity scores (n_docs,).
        """
        return similarity_search_faiss(
            query_features,
            corpus_features,
            distance_metric=self.config.distance_fn,
            postprocess=self.config.scale_dist,
        )
