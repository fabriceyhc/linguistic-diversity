# Troubleshooting

Installation and basic usage live in the [README](../README.md). This covers the
failure modes that are not obvious from an error message.

## A metric returns exactly the species count

If `TokenSemantics` on 30 tokens returns exactly `30.00`, or `DocumentSemantics` on 5
documents returns exactly `5.00`, the similarity matrix has collapsed to the identity
and the score is a saturation artifact, not a measurement. The library emits a
`RuntimeWarning` when it detects this.

The usual cause is a distance metric whose scale does not suit the embedding. L2 and L1
on transformer embeddings produce distances large enough that `exp(-d)` underflows to
zero. Use the defaults, or see the note in the README on `scale_dist`, which is only
meaningful with `cosine`.

## CUDA is "available" but every allocation fails

`torch.cuda.is_available()` returning `True` only means a driver and device were found.
On WSL in particular, allocation can still fail with
`CUDA-capable device(s) is/are busy or unavailable`.

The library probes CUDA with a real allocation before selecting it and falls back to CPU
with a warning, so metrics keep working. To fix the GPU itself:

- Check that the torch build matches the driver: `torch.version.cuda` against
  `nvidia-smi`. A torch built for CUDA 13 will not run on a CUDA 12 driver.
- If `cuCtxCreate` succeeds but torch fails at stream creation, the torch version is too
  new for the driver. Installing an earlier torch built for your CUDA version resolves
  it — this is more common than a genuinely broken driver.
- Libraries with custom modelling code (benepar, some embedding checkpoints) select
  their own device and bypass the fallback. `CUDA_VISIBLE_DEVICES=""` forces CPU.

To disable GPU explicitly: `TokenSemantics({'use_cuda': False})`.

## Out of memory when sweeping models

Models are cached indefinitely so repeated metric construction is cheap, which is the
wrong trade-off when evaluating many checkpoints in one process. Call
`clear_model_cache()` between them:

```python
from linguistic_diversity import DocumentSemantics, clear_model_cache

for name in model_names:
    metric = DocumentSemantics({'model_name': name})
    ...
    clear_model_cache()
```

## Optional dependencies

Some metrics need packages that cannot always be installed:

| Package | Needed for | Note |
|---|---|---|
| `benepar` | `ConstituencyParse` | `pip install '.[syntactic]'`; downloads a ~65MB model |
| `g2p-en`, `pyphen`, `pronouncing` | `Phonemic`, `Rhythmic` | `pip install '.[phonological]'` |
| `karateclub` | `ldp` / `feather` similarity | Pins `numpy<1.23`, so it conflicts with this package. The default `tree_edit_distance` needs nothing extra and discriminates better. |

Tests for these are skipped rather than failed when the dependency is absent.

## Missing NLTK or spaCy data

Phonemic and syntactic metrics need corpora that are not bundled:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords cmudict averaged_perceptron_tagger_eng punkt
```

Note the tagger is `averaged_perceptron_tagger_eng` on modern NLTK. The older
`averaged_perceptron_tagger` name will not satisfy `g2p_en`, which fails by returning
empty phoneme strings rather than raising — so the symptom is a silently wrong score,
not an error.

## macOS hangs on the first model load

On Apple silicon, constructing a semantic metric can hang indefinitely instead of
raising. The process sits at 100% CPU with no output and does not respond to Ctrl-C,
because it is stuck below Python in native code.

The likely cause is two OpenMP runtimes in one process. Both wheels bundle their own
copy:

```
torch/lib/libomp.dylib          (torch, macosx_arm64)
faiss/.dylibs/libomp.dylib      (faiss-cpu, macosx_arm64)
```

Linux resolves this to a single shared `libgomp` and Windows does not use OpenMP the
same way, which fits the observation that only macOS hangs. This has not been confirmed
against a stack trace, so treat it as the leading explanation rather than a diagnosis.
Working around it:

```bash
export OMP_NUM_THREADS=1        # serialises the OpenMP regions
export KMP_DUPLICATE_LIB_OK=TRUE
```

macOS is not covered by CI for this reason, so regressions there will not be caught
before a release. If you develop on a Mac and want the matrix restored, the exclusion
is a single line in `.github/workflows/ci.yml`.

## Large corpora are slow

Exact diversity builds an O(n²) similarity matrix. Use `estimate_diversity()`, which
samples at increasing sizes and extrapolates. Raising `batch_size` helps on GPU, and
`n_components` enables PCA before the similarity computation.
