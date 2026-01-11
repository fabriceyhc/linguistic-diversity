"""Comprehensive example demonstrating all diversity metrics."""

import pandas as pd
from linguistic_diversity import (
    # Semantic
    TokenSemantics,
    DocumentSemantics,
    # Syntactic
    DependencyParse,
    ConstituencyParse,
    # Morphological
    PartOfSpeechSequence,
    # Phonological
    Rhythmic,
    # Phonemic (optional - uses g2p_en by default)
)

try:
    from linguistic_diversity import Phonemic
    # Try to import g2p_en to check if phonemic will work
    try:
        from g2p_en import G2p  # noqa: F401
        PHONEMIC_AVAILABLE = True
    except ImportError:
        PHONEMIC_AVAILABLE = False
except ImportError:
    PHONEMIC_AVAILABLE = False


def main() -> None:
    """Run comprehensive diversity analysis across all dimensions."""
    # Example corpora with different diversity characteristics
    print("=" * 70)
    print("Linguistic Diversity - Comprehensive Analysis")
    print("=" * 70)

    corpus1 = [
        "one massive earth",
        "an enormous globe",
        "the colossal world",
    ]
    corpus1_name = "High Paraphrase"

    corpus2 = [
        "basic human right",
        "you were right",
        "make a right",
    ]
    corpus2_name = "Low Diversity"

    # Collect results
    results = []

    # ================================================================
    # SEMANTIC DIVERSITY
    # ================================================================
    print("\n" + "=" * 70)
    print("1. SEMANTIC DIVERSITY")
    print("=" * 70)

    print("\n  a) Token-Level Semantics (contextualized embeddings)")
    print("  " + "-" * 66)
    metric = TokenSemantics({"use_cuda": False, "verbose": False})

    div1 = metric(corpus1)
    div2 = metric(corpus2)
    results.append(
        {
            "Dimension": "Semantic",
            "Metric": "Token Semantics",
            corpus1_name: f"{div1:.2f}",
            corpus2_name: f"{div2:.2f}",
        }
    )
    print(f"    {corpus1_name}: {div1:.2f}")
    print(f"    {corpus2_name}:   {div2:.2f}")

    print("\n  b) Document-Level Semantics (sentence embeddings)")
    print("  " + "-" * 66)
    metric = DocumentSemantics(
        {"model_name": "all-MiniLM-L6-v2", "use_cuda": False, "verbose": False}
    )

    div1 = metric(corpus1)
    div2 = metric(corpus2)
    results.append(
        {
            "Dimension": "Semantic",
            "Metric": "Document Semantics",
            corpus1_name: f"{div1:.2f}",
            corpus2_name: f"{div2:.2f}",
        }
    )
    print(f"    {corpus1_name}: {div1:.2f}")
    print(f"    {corpus2_name}:   {div2:.2f}")

    # ================================================================
    # SYNTACTIC DIVERSITY
    # ================================================================
    print("\n" + "=" * 70)
    print("2. SYNTACTIC DIVERSITY")
    print("=" * 70)

    print("\n  a) Dependency Parse Trees (LDP embeddings)")
    print("  " + "-" * 66)
    metric = DependencyParse({"similarity_type": "ldp", "verbose": False})

    div1 = metric(corpus1)
    div2 = metric(corpus2)
    results.append(
        {
            "Dimension": "Syntactic",
            "Metric": "Dependency Parse",
            corpus1_name: f"{div1:.2f}",
            corpus2_name: f"{div2:.2f}",
        }
    )
    print(f"    {corpus1_name}: {div1:.2f}")
    print(f"    {corpus2_name}:   {div2:.2f}")

    print("\n  b) Constituency Parse Trees (phrase structure)")
    print("  " + "-" * 66)
    try:
        metric = ConstituencyParse({"similarity_type": "ldp", "verbose": False})

        div1 = metric(corpus1)
        div2 = metric(corpus2)
        results.append(
            {
                "Dimension": "Syntactic",
                "Metric": "Constituency Parse",
                corpus1_name: f"{div1:.2f}",
                corpus2_name: f"{div2:.2f}",
            }
        )
        print(f"    {corpus1_name}: {div1:.2f}")
        print(f"    {corpus2_name}:   {div2:.2f}")
    except Exception as e:
        print(f"    Skipped: {str(e)[:60]}")

    # ================================================================
    # MORPHOLOGICAL DIVERSITY
    # ================================================================
    print("\n" + "=" * 70)
    print("3. MORPHOLOGICAL DIVERSITY")
    print("=" * 70)

    print("\n  a) Part-of-Speech Sequences (POS tag alignment)")
    print("  " + "-" * 66)
    metric = PartOfSpeechSequence({"verbose": False})

    div1 = metric(corpus1)
    div2 = metric(corpus2)
    results.append(
        {
            "Dimension": "Morphological",
            "Metric": "POS Sequence",
            corpus1_name: f"{div1:.2f}",
            corpus2_name: f"{div2:.2f}",
        }
    )
    print(f"    {corpus1_name}: {div1:.2f}")
    print(f"    {corpus2_name}:   {div2:.2f}")

    # ================================================================
    # PHONOLOGICAL DIVERSITY
    # ================================================================
    print("\n" + "=" * 70)
    print("4. PHONOLOGICAL DIVERSITY")
    print("=" * 70)

    print("\n  a) Rhythmic Patterns (stress and weight)")
    print("  " + "-" * 66)
    try:
        metric = Rhythmic({"verbose": False})

        div1 = metric(corpus1)
        div2 = metric(corpus2)
        results.append(
            {
                "Dimension": "Phonological",
                "Metric": "Rhythmic",
                corpus1_name: f"{div1:.2f}",
                corpus2_name: f"{div2:.2f}",
            }
        )
        print(f"    {corpus1_name}: {div1:.2f}")
        print(f"    {corpus2_name}:   {div2:.2f}")
    except ImportError:
        print("    Skipped: cadences library not installed")
        print("    Install with: pip install cadences")

    print("\n  b) Phonemic Sequences (IPA phonemes)")
    print("  " + "-" * 66)
    if PHONEMIC_AVAILABLE:
        try:
            metric = Phonemic({"verbose": False})

            div1 = metric(corpus1)
            div2 = metric(corpus2)
            results.append(
                {
                    "Dimension": "Phonological",
                    "Metric": "Phonemic",
                    corpus1_name: f"{div1:.2f}",
                    corpus2_name: f"{div2:.2f}",
                }
            )
            print(f"    {corpus1_name}: {div1:.2f}")
            print(f"    {corpus2_name}:   {div2:.2f}")
        except Exception as e:
            print(f"    Skipped: {str(e)[:60]}")
    else:
        print("    Skipped: g2p_en library not installed")
        print("    Install with: pip install linguistic-diversity[phonological]")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print(
        f"""
The '{corpus1_name}' corpus contains paraphrases with similar meanings,
while '{corpus2_name}' has the word 'right' used in different contexts.

- Token/Document Semantics: Should show low diversity for {corpus1_name}
  (similar meanings despite different words)

- Syntactic: May show different diversity based on structural similarity

- Morphological: Depends on POS sequence similarity

- Phonological: Depends on rhythmic and phonemic patterns

Diversity scores represent the "effective number" of distinct linguistic units.
For example, a score of 2.5 means the corpus has the diversity equivalent
to 2.5 completely distinct items.
"""
    )


if __name__ == "__main__":
    main()
