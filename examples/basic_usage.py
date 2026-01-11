"""Basic usage examples for linguistic diversity metrics."""

from linguistic_diversity import TokenSemantics, DocumentSemantics


def main() -> None:
    """Run basic diversity examples."""
    # Example corpora with different diversity characteristics
    high_paraphrase = [
        "one massive earth",
        "an enormous globe",
        "the colossal world",
    ]

    low_diversity = [
        "basic human right",
        "you were right",
        "make a right",
    ]

    print("=" * 60)
    print("Linguistic Diversity - Basic Examples")
    print("=" * 60)

    # Token-level semantic diversity
    print("\n1. Token-Level Semantic Diversity")
    print("-" * 60)

    token_metric = TokenSemantics({
        "model_name": "bert-base-uncased",
        "use_cuda": False,  # Set to True if you have a GPU
        "verbose": False,
    })

    div1 = token_metric(high_paraphrase)
    div2 = token_metric(low_diversity)

    print(f"High paraphrase corpus: {div1:.2f}")
    print(f"Low diversity corpus:   {div2:.2f}")
    print(f"\nInterpretation: The high-paraphrase corpus has effective")
    print(f"               {div1:.0f} distinct token meanings")

    # Document-level semantic diversity
    print("\n2. Document-Level Semantic Diversity")
    print("-" * 60)

    doc_metric = DocumentSemantics({
        "model_name": "all-MiniLM-L6-v2",  # Smaller, faster model
        "use_cuda": False,
        "verbose": False,
    })

    div1 = doc_metric(high_paraphrase)
    div2 = doc_metric(low_diversity)

    print(f"High paraphrase corpus: {div1:.2f}")
    print(f"Low diversity corpus:   {div2:.2f}")
    print(f"\nInterpretation: Despite being paraphrases, the high-paraphrase")
    print(f"               corpus has {div1:.1f} distinct document meanings")

    # Ranking by similarity
    print("\n3. Document Ranking by Similarity")
    print("-" * 60)

    query = ["the planet we live on"]
    ranking, scores = doc_metric.rank_similarity(query, high_paraphrase + low_diversity, top_n=3)

    print(f"Query: {query[0]}")
    print(f"\nTop 3 most similar documents:")
    for i, (doc, score) in enumerate(zip(ranking, scores), 1):
        print(f"  {i}. [{score:.3f}] {doc}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
