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


# Similarity floors measured on REFERENCE_CORPUS, keyed by (metric class, model).
# Shipping them avoids an encode on first use for the common cases; anything else
# is calibrated on demand and cached for the process lifetime.
# Measured on REFERENCE_CORPUS, which is what auto-calibration would compute, so
# the shipped and on-demand paths agree. Note these run lower than the same
# encoders' baseline on same-domain text (mpnet 0.053 here against 0.075 on
# cross-prompt McDiv responses): the reference sentences share no topic, register
# or length, while responses from one generation setup share style even when their
# content is unrelated. The lower estimate is the deliberate choice -- it
# under-corrects, whereas too high a floor would drive genuinely related documents
# to zero similarity and inflate diversity.
KNOWN_SIMILARITY_FLOORS: dict[tuple[str, str], float] = {
    ("DocumentSemantics", "all-mpnet-base-v2"): 0.053,
    ("DocumentSemantics", "sentence-transformers/all-mpnet-base-v2"): 0.053,
    ("DocumentSemantics", "all-MiniLM-L6-v2"): 0.058,
    ("DocumentSemantics", "sentence-transformers/all-MiniLM-L6-v2"): 0.058,
    ("DocumentSemantics", "BAAI/bge-large-en-v1.5"): 0.351,
    ("DocumentSemantics", "BAAI/bge-base-en-v1.5"): 0.377,
    ("TokenSemantics", "bert-base-uncased"): 0.117,
}

_FLOOR_CACHE: dict[tuple[str, str], float] = {}


def clear_floor_cache() -> None:
    """Forget every auto-calibrated similarity floor."""
    _FLOOR_CACHE.clear()


def _resolve_similarity_floor(metric: Any) -> float | None:
    """Return the floor to apply, calibrating on the reference corpus if needed.

    Resolution order: an explicit float is used as given; None disables the
    correction; "auto" consults KNOWN_SIMILARITY_FLOORS, then the process cache,
    and otherwise measures the encoder's baseline on REFERENCE_CORPUS -- a fixed,
    shipped set of mutually unrelated sentences, never the corpus being scored.

    The median is used rather than the mean so that a typical unrelated pair lands
    exactly at zero, and so that a handful of accidentally-related reference pairs
    cannot drag the estimate.
    """
    configured = metric.config.similarity_floor
    if configured is None:
        return None
    if isinstance(configured, (int, float)):
        return float(configured)
    if configured != "auto":
        raise ValueError(f"similarity_floor must be a float, None, or 'auto'; got {configured!r}")

    key = (type(metric).__name__, metric.config.model_name)
    if key in _FLOOR_CACHE:
        return _FLOOR_CACHE[key]
    if key in KNOWN_SIMILARITY_FLOORS:
        _FLOOR_CACHE[key] = KNOWN_SIMILARITY_FLOORS[key]
        return _FLOOR_CACHE[key]

    from ..reference import REFERENCE_CORPUS

    # Bypass the floor while measuring it, or this recurses.
    metric._calibrating = True
    try:
        features, _species = metric.extract_features(list(REFERENCE_CORPUS))
        Z = np.asarray(metric.calculate_similarities(features), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - never let calibration break scoring
        warnings.warn(
            f"Could not calibrate a similarity floor for {key[1]!r} ({exc}); "
            "proceeding without the correction. Scores will be capped below the "
            "effective-number interpretation. Pass similarity_floor explicitly to "
            "silence this.",
            RuntimeWarning,
            stacklevel=3,
        )
        _FLOOR_CACHE[key] = 0.0
        return 0.0
    finally:
        metric._calibrating = False

    off_diagonal = Z[~np.eye(Z.shape[0], dtype=bool)]
    floor = float(np.clip(np.median(off_diagonal), 0.0, 0.95))
    _FLOOR_CACHE[key] = floor
    return floor


def _load_cross_encoder(model_name: str, device: Any) -> Any:
    """Load and cache a cross-encoder, alongside the bi-encoders."""
    from sentence_transformers import CrossEncoder

    cache_key = f"CrossEncoder:{model_name}:device={device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = CrossEncoder(model_name, device=str(device))
    return _MODEL_CACHE[cache_key]


def _entailment_index(model: Any) -> int | None:
    """Which output is 'entailment', or None if this is a graded-similarity model.

    Detected from the model's own label map rather than assumed, because the three
    NLI classes are not in a consistent order across checkpoints.
    """
    if getattr(model.config, "num_labels", 1) <= 1:
        return None
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).lower().startswith("entail"):
            return int(idx)
    raise ValueError(
        f"cross_encoder has {model.config.num_labels} outputs but no 'entailment' "
        f"label among {list(id2label.values())}. Use a graded similarity model "
        "(single output, e.g. cross-encoder/stsb-roberta-large) or an NLI model "
        "whose config declares an entailment class."
    )


def cross_encoder_similarities(
    corpus: list[str],
    model: Any,
    batch_size: int = 64,
) -> npt.NDArray[np.float64]:
    """Similarity matrix from a cross-encoder, symmetrised over both orderings.

    Every unordered pair is scored twice, once in each direction, and averaged --
    cross-encoders are not symmetric, and a diversity measure must be. Graded models
    contribute their score directly; NLI models contribute the entailment probability.

    Returns a matrix in [0, 1] with a unit diagonal, which is what the Hill number and
    the spectral index both require. The diagonal is set rather than measured: a
    document is identical to itself by definition, and asking the model costs n passes
    to be told so approximately.
    """
    n = len(corpus)
    Z = np.eye(n, dtype=np.float64)
    if n < 2:
        return Z

    pairs: list[tuple[str, str]] = []
    slots: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            slots.append((i, j))
            pairs.append((corpus[i], corpus[j]))
            pairs.append((corpus[j], corpus[i]))

    entail = _entailment_index(model)
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        **({"apply_softmax": True} if entail is not None else {}),
    )
    scores = np.asarray(scores, dtype=np.float64)
    values = scores[:, entail] if entail is not None else scores.ravel()

    for k, (i, j) in enumerate(slots):
        v = float(np.clip(0.5 * (values[2 * k] + values[2 * k + 1]), 0.0, 1.0))
        Z[i, j] = Z[j, i] = v

    # Identical strings are identical, whatever the model says. Cross-encoders score
    # a sentence against itself at ~0.96 rather than 1.0, and that gap is not small:
    # four byte-identical documents come out at 1.149 effective species instead of 1.
    # A bi-encoder never has this problem, since identical text gives identical
    # vectors and cosine 1 exactly.
    groups: dict[str, list[int]] = {}
    for idx, text in enumerate(corpus):
        groups.setdefault(text, []).append(idx)
    for members in groups.values():
        if len(members) > 1:
            block = np.ix_(members, members)
            Z[block] = 1.0
    return Z


def _apply_similarity_floor(
    Z: npt.NDArray[np.float64], floor: float | None
) -> npt.NDArray[np.float64]:
    """Rescale so that ``floor`` maps to 0 and 1 stays 1.

    Monotone, so no corpus ordering changes; pinned at both ends, so identical
    items keep similarity 1 and replication invariance survives; and dependent
    only on the pair and a constant, so no corpus-level statistic leaks in.
    """
    if floor is None:
        return Z
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"similarity_floor must be in [0, 1), got {floor}")
    if floor == 0.0:
        return Z
    rescaled = (Z - floor) / (1.0 - floor)
    np.clip(rescaled, 0.0, 1.0, out=rescaled)
    np.fill_diagonal(rescaled, 1.0)
    return rescaled


@dataclass
class SemanticConfig(MetricConfig):
    """Configuration for semantic diversity metrics."""

    # Similarity computation
    distance_fn: int = faiss.METRIC_INNER_PRODUCT
    scale_dist: str | None = None
    # Rescale similarity so that "unrelated" maps to 0 rather than to the encoder's
    # floor:  z' = max(0, (z - floor) / (1 - floor)).
    #
    # Sentence encoders do not send unrelated text to orthogonal vectors; they send
    # it to cosine ~0.11 (mpnet) to ~0.46 (bge-large). Because every document is
    # slightly similar to every *other* document, that floor accumulates: the
    # largest diversity any abundance can reach is n / (1 + (n-1)z), which tends to
    # 1/z as n grows. A floor of 0.46 caps a corpus at ~2.2 effective species
    # however large it is.
    #
    # The floor must be a constant, never estimated from the corpus at hand.
    # mean_adj subtracted the corpus's own mean and thereby made each pair's
    # similarity depend on unrelated documents, which cost replication invariance
    # and left two identical items scoring 0.78. A fixed constant keeps the
    # transform pair-local, monotone, and fixed at both ends: z=1 maps to 1, and
    # z<=floor maps to 0.
    #
    # "auto" looks the floor up for this (metric, encoder) pair and calibrates it
    # on REFERENCE_CORPUS if unknown, caching the result. A float sets it
    # explicitly; None disables the correction and restores pre-1.0.3 behaviour.
    # See benchmarks/embedder_selection/calibrate_floor.py.
    similarity_floor: float | str | None = "auto"

    power_reg: bool = False
    # Off by default. mean_adj subtracts the off-diagonal mean from every
    # off-diagonal entry, so two *identical* items no longer score 1.0 while the
    # diagonal still does -- which breaks replication invariance. It also scores
    # worse against human judgments. See benchmarks/embedder_selection/.
    mean_adj: bool = False

    # Feature processing
    remove_stopwords: bool = False
    remove_punct: bool = False
    n_components: int | str | None = None  # PCA dimensions ("auto" or int)

    # Score pairs with a cross-encoder instead of comparing bi-encoder embeddings.
    #
    # A bi-encoder reads each document once and compares vectors, so "unrelated" lands
    # on the encoder's floor rather than on 0 and the cap above applies. A cross-encoder
    # reads both documents together and is trained to output the comparison directly,
    # so unrelated text scores ~0.01 and there is no floor to correct.
    #
    # Recommended: "cross-encoder/stsb-roberta-large", trained for *graded* similarity,
    # which is what a similarity matrix wants. Measured on 1,270 human-scored sets it
    # matches the best embedding pipeline while needing no similarity floor, no
    # hubness correction and no prompt. See benchmarks/similarity_transforms/.
    #
    # NLI cross-encoders also work and are auto-detected: the entailment probability
    # is used, symmetrised over both orderings. Prefer a model whose *neutral* class
    # is well calibrated -- "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    # -- because SNLI/MNLI-only models label unrelated text "contradiction" and the
    # entailment reading then depends on an artefact of that training data.
    #
    # The cost is the reason this is opt-in rather than the default: O(n^2) forward
    # passes against O(n) encodes, about 45x a bi-encoder at 40 documents and
    # quadratic thereafter.
    # Set per class, not here: DocumentSemantics defaults to a cross-encoder,
    # TokenSemantics cannot use one because its species are tokens, not documents.
    cross_encoder: str | None = None
    cross_encoder_batch_size: int = 64
    # Refuse rather than hang: n documents cost n*(n-1) forward passes. At the
    # measured 170 passes/s this limit is roughly 25 minutes.
    cross_encoder_max_docs: int = 512
    # Warn above this, since the cost is quadratic and easy to walk into unawares.
    cross_encoder_warn_docs: int = 64

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
        # to rank agreement with ground truth. See benchmarks/embedder_selection/.
        return {
            "model_name": "bert-base-uncased",
            "batch_size": 16,
            "use_cuda": True,
            "distance_fn": faiss.METRIC_INNER_PRODUCT,
            "scale_dist": None,
            # mean_adj is off: it cost both correctness and accuracy. See
            # benchmarks/embedder_selection/ablate_similarity.py.
            "mean_adj": False,
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

        if not getattr(self, "_calibrating", False):
            Z = _apply_similarity_floor(Z, _resolve_similarity_floor(self))

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

        # Optional cross-encoder kernel. The bi-encoder is still loaded: its
        # embeddings remain the features, so ranking and selection keep working.
        self.cross_encoder = (
            _load_cross_encoder(self.config.cross_encoder, self.device)
            if self.config.cross_encoder
            else None
        )
        self._pair_corpus: list[str] | None = None

        # The cross-encoder path never consults the floor, so validate it here rather
        # than let a bad or pointless value pass silently.
        floor = self.config.similarity_floor
        if isinstance(floor, (int, float)) and not 0.0 <= float(floor) < 1.0:
            raise ValueError(f"similarity_floor must be in [0, 1), got {floor}")
        if self.cross_encoder is not None and floor is not None and floor != "auto":
            warnings.warn(
                f"similarity_floor={floor!r} is ignored when cross_encoder is set: a "
                "cross-encoder puts unrelated text at ~0.01, so there is no floor to "
                "subtract. Pass cross_encoder=None to use the bi-encoder and the floor.",
                RuntimeWarning,
                stacklevel=2,
            )

    @classmethod
    def _config_class(cls) -> type[SemanticConfig]:
        return SemanticConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        # all-mpnet-base-v2 is a deliberate choice, not an oversight, and the two
        # halves of benchmarks/embedder_selection/ disagree about it.
        #
        #   calibration      mpnet 0.97, bge-large 0.58  (is the reported number
        #                    of concepts numerically right?)
        #   human agreement  mpnet +0.581, bge-large +0.779  (are corpora ordered
        #                    the way people order them?)
        #
        # The default optimises calibration: a score read as a quantity should
        # mean what it says, and mpnet is a third of the size. **If you are
        # comparing corpora against each other -- the more common case -- set
        # model_name="BAAI/bge-large-en-v1.5", which the benchmark recommends.**
        #
        # mpnet's cosines do fall below zero on about 1.35% of McDiv response
        # pairs, which is why compute_similarity_matrix_faiss clamps to [0, 1];
        # bge-large never goes below +0.32 and never needs it.
        # The cross-encoder is the default because it is markedly better on both
        # criteria that decide anything: held-out human agreement 0.709 -> 0.816
        # and calibration ratio 0.986 -> 0.9998, with no similarity floor, no
        # hubness correction and no prompt required.
        #
        # It is also **quadratic**, and that is not a small constant. Measured on
        # an RTX 2060 at ~170 forward passes/s:
        #
        #     n=10    0.5s      n=100    55s      n=500   ~25 min
        #     n=25    3.2s      n=200   3.9 min   n=1000  ~1.6 h
        #
        # against 0.1-0.2s for the bi-encoder at any of those sizes. Pass
        # cross_encoder=None to restore the bi-encoder, which is what to do for
        # corpora of more than a few hundred documents, for estimate_diversity,
        # and for anything that scores repeatedly.
        return {
            "model_name": "all-mpnet-base-v2",
            "cross_encoder": "cross-encoder/stsb-roberta-large",
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

        # A cross-encoder scores text pairs, not vectors, so the documents have to
        # reach calculate_similarities. Held here rather than threaded through the
        # feature type, which every other metric shares.
        if self.cross_encoder is not None:
            self._pair_corpus = list(corpus)

        return embeddings, corpus

    def calculate_similarities(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate pairwise document similarities.

        Args:
            features: Document embeddings (n_docs x dim).

        Returns:
            Similarity matrix (n_docs x n_docs).

        Raises:
            ValueError: In cross-encoder mode, if ``extract_features`` was not called
                for this corpus, or if the corpus exceeds ``cross_encoder_max_docs``.
        """
        if self.cross_encoder is not None:
            return self._cross_encoder_similarities(features)

        Z = compute_similarity_matrix_faiss(
            features,
            distance_metric=self.config.distance_fn,
            postprocess=self.config.scale_dist,
        )

        if not getattr(self, "_calibrating", False):
            Z = _apply_similarity_floor(Z, _resolve_similarity_floor(self))

        if self.config.power_reg:
            Z = np.power(Z, 2)

        if self.config.mean_adj:
            off_diag_mask = ~np.eye(Z.shape[0], dtype=bool)
            mean_sim = Z[off_diag_mask].mean()
            Z[off_diag_mask] -= mean_sim
            Z = np.maximum(Z, 0)

        return Z

    def _cross_encoder_similarities(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Pairwise scores from the configured cross-encoder.

        No similarity floor is applied. The floor exists because a bi-encoder puts
        unrelated text at cosine 0.05-0.46 rather than 0; a cross-encoder puts it at
        about 0.01, so there is nothing to subtract and subtracting anyway would only
        add a second, unvalidated constant.
        """
        corpus = self._pair_corpus
        n = len(features)
        if corpus is None or len(corpus) != n:
            raise ValueError(
                "cross-encoder mode needs the documents themselves, which are captured "
                "by extract_features. Call extract_features(corpus) immediately before "
                "calculate_similarities, or use diversity()/diversity_profile(), which "
                "do that for you."
            )
        limit = self.config.cross_encoder_max_docs
        passes = n * (n - 1)
        if n > limit:
            raise ValueError(
                f"the cross-encoder would need {passes:,} forward passes for {n} "
                f"documents (roughly {passes / 170 / 60:.0f} minutes), above the "
                f"cross_encoder_max_docs limit of {limit}. Pass cross_encoder=None "
                "for the bi-encoder, which is O(n) encodes and the right choice at "
                "this size, or raise cross_encoder_max_docs deliberately."
            )
        if n > self.config.cross_encoder_warn_docs:
            warnings.warn(
                f"scoring {n} documents with a cross-encoder needs {passes:,} forward "
                f"passes, roughly {passes / 170:.0f}s. The cost is quadratic; pass "
                "cross_encoder=None for the bi-encoder if that is too slow.",
                RuntimeWarning,
                stacklevel=3,
            )
        return cross_encoder_similarities(
            corpus, self.cross_encoder, batch_size=self.config.cross_encoder_batch_size
        )

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
