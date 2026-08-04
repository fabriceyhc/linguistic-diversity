"""External diversity metrics, implemented here for comparison.

These are not part of the library. They live in the benchmarks because a claim
that similarity-sensitive Hill numbers are worth using has to be measured against
the alternatives, on the same corpora, with the same similarity matrix.

Currently: the Vendi Score (Friedman & Dieng, TMLR 2023).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def vendi_score(K: npt.NDArray[np.float64], q: float = 1.0) -> float:
    """Vendi Score: the exponential Shannon entropy of the eigenvalues of K/n.

    Friedman & Dieng, *The Vendi Score: A Diversity Evaluation Metric for Machine
    Learning* (TMLR 2023), arXiv:2210.02410.

        VS(x_1..x_n) = exp(-sum_i lambda_i log lambda_i)

    where lambda are the eigenvalues of K/n. K must be symmetric, positive
    semi-definite, and have unit diagonal, which makes the eigenvalues sum to 1
    and lets them be read as a probability distribution.

    It is the closest published relative of what this library computes, and the
    comparison is worth stating precisely. Both return an effective number of
    elements from a similarity matrix, and both agree at the extremes -- K = I
    gives n, K all-ones gives 1. They differ in between:

    - Vendi reads diversity off the **eigenvalue spectrum** of K/n, so abundance
      enters only through which items are in the matrix. Leinster-Cobbold carries
      an explicit abundance vector p, and D_q = (sum_i p_i (Zp)_i^(q-1))^(1/(1-q))
      responds to how often each species occurs.
    - Vendi requires K positive semi-definite, because a negative eigenvalue makes
      log(lambda) undefined. Leinster-Cobbold requires only Z in [0, 1] with unit
      diagonal, which is a weaker condition -- clamping or rescaling a similarity
      matrix can break positive semi-definiteness while leaving the Hill number
      perfectly well defined.

    Args:
        K: Similarity matrix (n x n), symmetric with unit diagonal.
        q: Order parameter. q=1 is the published Vendi Score; other orders follow
            the Renyi generalisation used in later work.

    Returns:
        Effective number of distinct elements.

    Raises:
        ValueError: If K has a materially negative eigenvalue, which means it is
            not a valid kernel and the score is undefined rather than merely
            inaccurate.
    """
    n = K.shape[0]
    if n == 0:
        return 0.0
    sym = (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T) / 2
    eigenvalues = np.linalg.eigvalsh(sym / n)

    if eigenvalues.min() < -1e-6:
        raise ValueError(
            f"K is not positive semi-definite (smallest eigenvalue "
            f"{eigenvalues.min():.3e}); the Vendi Score is undefined for it."
        )
    # Clip the numerical dust that eigvalsh leaves around zero.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    eigenvalues = eigenvalues / total
    nonzero = eigenvalues[eigenvalues > 0]

    if q == 1.0:
        return float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    if np.isinf(q):
        return float(1.0 / nonzero.max())
    return float(np.power(np.sum(np.power(nonzero, q)), 1.0 / (1.0 - q)))


class VendiWrapper:
    """Score a corpus with Vendi, reusing another metric's similarity matrix.

    Holding the similarity function fixed is the point: it isolates the choice of
    diversity index from the choice of representation, which is the only way the
    comparison says anything about the index.
    """

    def __init__(self, base_metric: Any, q: float = 1.0) -> None:
        self.base = base_metric
        self.q = q

    def __call__(self, corpus: list[str]) -> float:
        if not corpus:
            return 0.0
        if len(corpus) == 1:
            return 1.0
        features, _species = self.base.extract_features(corpus)
        K = np.asarray(self.base.calculate_similarities(features), dtype=np.float64)
        return vendi_score(K, q=self.q)

    def extract_features(self, corpus: list[str]) -> Any:
        return self.base.extract_features(corpus)

    def calculate_similarities(self, features: Any) -> Any:
        return self.base.calculate_similarities(features)


# ---------------------------------------------------------------------------------
# Decan: progressive conditional surprise
# ---------------------------------------------------------------------------------


class Decan:
    """Diversity as how much surprise survives in-context learning.

    Khoriaty, Williams-King & Feng, *"I've Seen How This Goes": Characterizing the
    Diversity of LLM Generations and Human Writing via Progressive Conditional
    Surprise* (arXiv:2606.01811, ICML 2026 workshop).

    A different paradigm from everything else here. There is no embedder, no
    similarity matrix and no reference corpus -- diversity is read off a base
    language model's per-token log-probabilities. Show the model k-1 responses, then
    ask how surprised it still is by the k-th. If the responses are alike, it has
    learned what it needs and the surprise collapses; if they are genuinely varied,
    it stays high.

        a_n  bits per byte of the last response, conditioned on all the others,
             averaged over random permutations so the score does not depend on
             which response happened to land last
        C    reciprocal of the geometric-mean per-byte perplexity of the responses
             scored individually -- without it, incoherent noise reads as maximally
             diverse
        D    C * a_n

    Relevant to this library precisely because it *cannot* have a similarity floor:
    it never computes a similarity. Whatever it gets wrong, it gets wrong for
    different reasons.

    Reimplemented from the paper's description rather than the authors' code, so
    treat absolute values as indicative and comparisons across corpora as the
    meaningful part.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        permutations: int = 8,
        seed: int = 20260803,
        device: str | None = None,
        max_tokens: int = 900,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.permutations = permutations
        self.rng = np.random.default_rng(seed)
        self.max_tokens = max_tokens
        self._sep = "\n\n"

    def _token_bits(self, texts: list[str]) -> tuple[list[float], list[int]]:
        """Total surprise in bits for each text, given everything before it.

        One forward pass over the concatenation. Returns per-text bits and the byte
        length each should be normalised by.
        """
        import torch

        ids: list[int] = []
        spans: list[tuple[int, int]] = []
        for i, text in enumerate(texts):
            piece = (self._sep if i else "") + text
            tokens = self.tokenizer.encode(piece)
            spans.append((len(ids), len(ids) + len(tokens)))
            ids += tokens
        ids = ids[: self.max_tokens]
        if len(ids) < 2:
            return [0.0] * len(texts), [max(len(t.encode()), 1) for t in texts]

        with torch.no_grad():
            tensor = torch.tensor([ids], device=self.device)
            logits = self.model(tensor).logits[0].float()
            logprobs = torch.log_softmax(logits[:-1], dim=-1)
            targets = tensor[0][1:]
            token_logprobs = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # token_logprobs[j] is the log-probability of ids[j+1].
        lp = token_logprobs.cpu().numpy()

        bits, byte_lens = [], []
        for text, (start, end) in zip(texts, spans, strict=True):
            end = min(end, len(ids))
            # The first token of the whole sequence has no prediction to score.
            lo, hi = max(start - 1, 0), max(end - 1, 0)
            segment = lp[lo:hi] if hi > lo else np.zeros(0)
            bits.append(float(-segment.sum() / np.log(2)))
            byte_lens.append(max(len(text.encode()), 1))
        return bits, byte_lens

    def __call__(self, corpus: list[str]) -> float:
        if len(corpus) < 2:
            return 0.0
        texts = [t for t in corpus if t and t.strip()]
        if len(texts) < 2:
            return 0.0

        # Coherence: per-byte perplexity of each response on its own.
        solo_bpb = []
        for text in texts:
            bits, byte_lens = self._token_bits([text])
            solo_bpb.append(bits[0] / byte_lens[0])
        coherence = float(2.0 ** (-np.mean(solo_bpb)))

        # a_n: bits per byte of the final response given all the others.
        finals = []
        for _ in range(self.permutations):
            order = self.rng.permutation(len(texts))
            ordered = [texts[i] for i in order]
            bits, byte_lens = self._token_bits(ordered)
            finals.append(bits[-1] / byte_lens[-1])
        return float(coherence * np.mean(finals))


# ---------------------------------------------------------------------------------
# PRDC: precision, recall, density, coverage
# ---------------------------------------------------------------------------------


def prdc(
    real: npt.NDArray[np.float64], fake: npt.NDArray[np.float64], k: int = 5
) -> dict[str, float]:
    """Fidelity and diversity against a *reference distribution*.

    Naeem, Oh, Uh, Choi & Yoo, *Reliable Fidelity and Diversity Metrics for
    Generative Models* (ICML 2020); precision/recall from Kynkaanniemi et al. (2019).

    The only entry here that measures diversity **relative to a reference** rather
    than internally. `recall` and `coverage` are the diversity halves: coverage asks
    what fraction of reference points have a generated point inside their k-NN ball,
    which is robust to the outliers that make recall unstable.

    A caveat that matters for how the results are read: these are distribution-level
    statistics designed for hundreds or thousands of samples. Applied to a five-item
    response set they are far outside their design regime, and any weak result there
    is a statement about sample size, not about the method.
    """
    if real.size == 0 or fake.size == 0:
        return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}
    k = max(1, min(k, len(real) - 1, len(fake) - 1)) if min(len(real), len(fake)) > 1 else 1

    def pairwise(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sqrt(
            np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T, 0.0)
        )

    def knn_radius(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        d = pairwise(x, x)
        np.fill_diagonal(d, np.inf)
        kk = min(k, d.shape[1] - 1) if d.shape[1] > 1 else 0
        return np.partition(d, kk, axis=1)[:, kk] if d.shape[1] > 1 else np.full(len(x), np.inf)

    r_radius = knn_radius(real)
    f_radius = knn_radius(fake)
    d_rf = pairwise(real, fake)  # (n_real, n_fake)

    within_real = d_rf <= r_radius[:, None]
    within_fake = d_rf <= f_radius[None, :]
    return {
        "precision": float(within_real.any(axis=0).mean()),
        "recall": float(within_fake.any(axis=1).mean()),
        "density": float(within_real.sum(axis=0).mean() / k),
        "coverage": float(within_real.any(axis=1).mean()),
    }
