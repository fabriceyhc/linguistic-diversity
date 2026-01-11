# Examples

This directory contains example scripts demonstrating the usage of linguistic diversity metrics.

## Basic Usage

Run the basic example:

```bash
python basic_usage.py
```

This demonstrates:
- Token-level semantic diversity
- Document-level semantic diversity
- Document ranking by similarity

## Requirements

Make sure you have installed the package:

```bash
pip install -e ..
```

Or with all dependencies:

```bash
pip install -e "..[all]"
```

## Performance Notes

- Set `use_cuda=True` in the config if you have a GPU for faster processing
- For large corpora, consider using `batch_size=64` or higher
- Models are cached after first load, so subsequent runs are faster
