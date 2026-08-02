# Development trail

These are the intermediate experiments that led to the final system, kept for
provenance. They are **superseded** — each is a standalone script that duplicates its
own setup, and none is imported by the pipeline in the parent directory.

Read them in roughly this order:

| script | what it established |
|---|---|
| `improved_analysis.py` | First pass beyond the zero-shot baseline; quick feature wins |
| `next_steps_implementation.py` | Added enhanced lexical metrics and inter-text similarity |
| `hybrid_stylometry.py` | Combined inter-text similarity with traditional stylometry |
| `validation_and_optimization.py` | Scaled to the full dataset, better models, confidence scoring |

The conclusions from all four are folded into `../full_dataset_evaluation.py` and
`../confidence_and_cv_evaluation.py`. Start there instead.
