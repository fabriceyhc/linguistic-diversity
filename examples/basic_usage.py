"""Basic usage examples for linguistic diversity metrics.

The two corpora are named for how they are *built*, not for how they score. Naming
a set "low diversity" up front would prejudge the very thing being measured -- and
in this case would be wrong, since the set with the repeated word is the more
semantically diverse of the two.
"""

from linguistic_diversity import DocumentSemantics, TokenSemantics, TypeTokenRatio


def main() -> None:
    """Run basic diversity examples."""
    # Every word unique, but all five sentences describe the same storm.
    all_unique_words = [
        "a violent tempest wrecked our village",
        "the fierce gale devastated their settlement",
        "that savage hurricane destroyed this community",
        "an intense cyclone flattened every township",
        "some brutal windstorm ruined nearby neighborhoods",
    ]

    # "run" five times, each in a different sense: jogging, managing, a tear in
    # fabric, executing a program, and a baseball point.
    one_word_many_senses = [
        "she went for a morning run",
        "he will run the entire company",
        "a run appeared in her stocking",
        "the program failed to run correctly",
        "they scored the winning run today",
    ]

    print("=" * 68)
    print("Linguistic Diversity - Basic Examples")
    print("=" * 68)
    print("\nBoth corpora: 5 documents, 30 words, 30 token species.")
    print("Only the content differs, so the scores are directly comparable.")

    # ----------------------------------------------------------------------
    print("\n1. What a lexical metric sees")
    print("-" * 68)

    ttr = TypeTokenRatio()
    print(f"    all unique words     type-token ratio {ttr(all_unique_words):.3f}")
    print(f"    one word many senses type-token ratio {ttr(one_word_many_senses):.3f}")
    print("\n    The first corpus scores a perfect 1.000: no word repeats anywhere.")

    # ----------------------------------------------------------------------
    print("\n2. Document-level semantic diversity")
    print("-" * 68)

    doc_metric = DocumentSemantics()
    unique_words_score = doc_metric(all_unique_words)
    many_senses_score = doc_metric(one_word_many_senses)

    print(f"    all unique words     {unique_words_score:.2f} of a possible 5")
    print(f"    one word many senses {many_senses_score:.2f} of a possible 5")
    print(
        f"\n    Reversed. The first corpus restates one idea five times, so it carries"
        f"\n    about {unique_words_score:.1f} distinct meanings despite its varied wording."
        f"\n    The second repeats 'run' but means something different each time."
    )

    # ----------------------------------------------------------------------
    print("\n3. Token-level semantic diversity")
    print("-" * 68)

    token_metric = TokenSemantics()
    print(f"    all unique words     {token_metric(all_unique_words):.2f} of a possible 30")
    print(f"    one word many senses {token_metric(one_word_many_senses):.2f} of a possible 30")
    print(
        "\n    Token level separates them far less: 'tempest', 'gale' and 'hurricane'"
        "\n    really are distinct as isolated words. The redundancy only becomes"
        "\n    visible once whole sentences are compared."
    )

    # ----------------------------------------------------------------------
    print("\n4. Ranking documents by similarity to a query")
    print("-" * 68)

    query = ["a hurricane destroyed the town"]
    ranking, scores = doc_metric.rank_similarity(
        query, all_unique_words + one_word_many_senses, top_n=3
    )

    print(f"    query: {query[0]}\n")
    for i, (doc, score) in enumerate(zip(ranking, scores), 1):
        print(f"    {i}. [{score:.3f}] {doc}")

    print("\n" + "=" * 68)
    print("Next: examples/all_metrics.py runs all four linguistic dimensions,")
    print("      or open examples/demo.ipynb for a guided tour.")
    print("=" * 68)


if __name__ == "__main__":
    main()
