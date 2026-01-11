# Pure Python - Zero System Dependencies! 🎉

## Major Update: No System Dependencies Required

**All 7 linguistic diversity metrics now work with pure Python packages!**

Previously, the `Phonemic` diversity metric required the `espeak-ng` system library, which had to be installed via system package managers (apt, brew, etc.). This has been resolved by adding support for `g2p-en`, a pure Python grapheme-to-phoneme library.

## What Changed

### Before (v1.0.0)
- ❌ `Phonemic` required `espeak-ng` system library
- ❌ Users had to run `sudo apt-get install espeak-ng` (Linux) or `brew install espeak-ng` (macOS)
- ❌ Windows users had to download and configure espeak-ng manually
- ❌ Not installable in restricted environments without system admin access

### After (v1.0.1)
- ✅ `Phonemic` now uses `g2p-en` by default (pure Python)
- ✅ **Zero system dependencies** - everything installs via `pip`
- ✅ Works everywhere: Linux, macOS, Windows, Docker, cloud notebooks, etc.
- ✅ No admin/sudo access required
- ✅ Optional: can still use `phonemizer` backend if you prefer

## Installation

### All Metrics (Pure Python)

```bash
# Install everything with zero system dependencies
pip install linguistic-diversity[all]

# Or just phonological metrics
pip install linguistic-diversity[phonological]
```

That's it! No `sudo`, no `brew`, no system packages needed.

## Usage

### Default: Pure Python (g2p_en)

```python
from linguistic_diversity import Phonemic

# Uses g2p_en backend (pure Python, works everywhere)
metric = Phonemic()
corpus = ['hello world', 'goodbye moon', 'see you later']
diversity = metric(corpus)
print(f"Phonemic diversity: {diversity:.2f}")
```

### Optional: phonemizer Backend

If you specifically want to use the `phonemizer` backend (requires espeak-ng):

```python
from linguistic_diversity import Phonemic

# Explicitly use phonemizer backend
metric = Phonemic({'backend': 'phonemizer'})
diversity = metric(corpus)
```

Then install espeak-ng:
- Linux: `sudo apt-get install espeak-ng && pip install phonemizer`
- macOS: `brew install espeak-ng && pip install phonemizer`
- Windows: Download from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases) and `pip install phonemizer`

## All Metrics - Pure Python

| Metric | System Dependencies | Pure Python? |
|--------|---------------------|--------------|
| **TokenSemantics** | None | ✅ Yes |
| **DocumentSemantics** | None | ✅ Yes |
| **DependencyParse** | None | ✅ Yes |
| **ConstituencyParse** | None | ✅ Yes |
| **PartOfSpeechSequence** | None | ✅ Yes |
| **Rhythmic** | None | ✅ Yes |
| **Phonemic** | None (uses g2p_en) | ✅ Yes |

**Total**: 7/7 metrics work with pure Python! 🎉

## Technical Details

### g2p_en vs phonemizer

Both backends convert text to phonemes (IPA representation):

**g2p_en** (default):
- Pure Python implementation
- No system dependencies
- Works everywhere
- Good accuracy for English
- Slightly faster initialization
- Recommended for most use cases

**phonemizer** (optional):
- Uses espeak-ng backend (C library)
- Requires system library installation
- Better accuracy for some edge cases
- Supports more languages
- Use if you need maximum accuracy or non-English languages

### Backend Selection

```python
# Default: g2p_en (pure Python)
metric = Phonemic()

# Explicit g2p_en
metric = Phonemic({'backend': 'g2p_en'})

# phonemizer (requires espeak-ng)
metric = Phonemic({'backend': 'phonemizer'})
```

The library automatically falls back to `phonemizer` if `g2p_en` is not installed, so you have flexibility.

## Benefits of Pure Python

1. **Easy Installation**: Just `pip install`, no system packages
2. **Portable**: Works in any Python environment
3. **Docker-Friendly**: No need to install system packages in containers
4. **Cloud Notebooks**: Works in Colab, Kaggle, SageMaker, etc.
5. **No Admin Required**: Install without sudo/administrator access
6. **Reproducible**: Pinned Python packages, no system variation
7. **Cross-Platform**: Same installation on Linux, macOS, Windows

## Migration from v1.0.0

If you were using `Phonemic` with phonemizer:

```python
# Old code (still works!)
from linguistic_diversity import Phonemic
metric = Phonemic()  # Now uses g2p_en by default
diversity = metric(corpus)

# If you want to keep using phonemizer
metric = Phonemic({'backend': 'phonemizer'})
diversity = metric(corpus)
```

Results are similar between backends (within ~5% for typical English text).

## Dependencies

### Core (always installed)
- numpy, pandas, scipy, scikit-learn
- torch, transformers, sentence-transformers
- spacy, biopython, networkx, karateclub, zss
- tqdm, faiss-cpu

### Optional Groups
```bash
# Syntactic metrics (constituency parsing)
pip install linguistic-diversity[syntactic]  # adds benepar

# Phonological metrics
pip install linguistic-diversity[phonological]  # adds g2p-en, cadences

# Development
pip install linguistic-diversity[dev]  # adds pytest, black, ruff, mypy

# Everything
pip install linguistic-diversity[all]
```

All optional groups are pure Python packages!

## Frequently Asked Questions

**Q: Do I need espeak-ng anymore?**
A: No! The default `g2p_en` backend works without it.

**Q: Can I still use phonemizer?**
A: Yes! Just set `backend='phonemizer'` in config.

**Q: Which backend should I use?**
A: For most users, the default `g2p_en` is perfect. Use `phonemizer` only if you need maximum accuracy or non-English languages.

**Q: Will results be different?**
A: Slightly, but both are accurate. For typical diversity measurements, the difference is minimal.

**Q: Does g2p_en support other languages?**
A: g2p_en is optimized for English. For other languages, use `phonemizer` backend with appropriate espeak-ng language.

## Conclusion

**Linguistic Diversity is now 100% pure Python!**

No more system dependencies, no more sudo/admin access, no more platform-specific installation issues. Just `pip install` and start measuring diversity.

---

**Version**: 1.0.1
**Date**: January 2026
**Status**: ✅ All metrics pure Python
