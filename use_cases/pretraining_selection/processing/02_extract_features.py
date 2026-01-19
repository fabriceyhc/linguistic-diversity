#!/usr/bin/env python3
"""Extract semantic and syntactic features from corpus documents.

This script:
1. Loads preprocessed corpus data
2. Extracts semantic features using sentence transformers
3. Extracts syntactic features using spaCy POS tagging
4. Saves features as memory-mapped arrays for efficient access
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import pickle
import numpy as np
import yaml
import torch
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
from tqdm import tqdm
import networkx as nx

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_documents(input_dir, dataset_name):
    """Load preprocessed documents.

    Args:
        input_dir: Input directory path
        dataset_name: Name of dataset

    Returns:
        List of document strings
    """
    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()
    train_file = input_dir / f"{clean_name}_train.pkl"

    print(f"   Loading: {train_file}")
    with open(train_file, 'rb') as f:
        documents = pickle.load(f)

    print(f"   ✓ Loaded {len(documents)} documents")
    return documents


def extract_semantic_features(documents, config):
    """Extract semantic features using sentence transformers.

    Args:
        documents: List of document strings
        config: Configuration dict

    Returns:
        numpy array of shape (n_documents, embedding_dim)
    """
    print(f"\n2. Extracting semantic features...")

    semantic_config = config['features']['semantic']
    model_name = semantic_config['model']
    batch_size = semantic_config['batch_size']
    use_gpu = semantic_config['use_gpu'] and torch.cuda.is_available()

    device = 'cuda' if use_gpu else 'cpu'
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")

    # Load model
    print(f"   Loading sentence transformer...")
    model = SentenceTransformer(model_name)
    if use_gpu:
        model = model.to(device)

    # Extract embeddings
    print(f"   Encoding {len(documents)} documents...")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True  # Normalize for cosine similarity
    )

    print(f"   ✓ Extracted embeddings: shape {embeddings.shape}")

    return embeddings


def extract_pos_tags(documents, spacy_model):
    """Extract POS tag sequences from documents.

    Args:
        documents: List of document strings
        spacy_model: Loaded spaCy model

    Returns:
        List of POS tag sequence strings
    """
    pos_sequences = []

    for doc in tqdm(documents, desc="   Extracting POS tags"):
        try:
            # Process with spaCy
            parsed = spacy_model(doc[:10000])  # Limit length for efficiency

            # Extract POS tags
            pos_tags = [token.pos_ for token in parsed if not token.is_space]

            # Join into string
            pos_sequence = ' '.join(pos_tags)
            pos_sequences.append(pos_sequence)

        except Exception as e:
            # On error, add empty sequence
            pos_sequences.append("")

    return pos_sequences


def extract_syntactic_features(documents, config):
    """Extract syntactic features using POS n-grams.

    Args:
        documents: List of document strings
        config: Configuration dict

    Returns:
        scipy sparse matrix of shape (n_documents, n_features)
    """
    print(f"\n3. Extracting syntactic features...")

    syntactic_config = config['features']['syntactic']
    spacy_model_name = syntactic_config['spacy_model']
    ngram_range = tuple(syntactic_config['ngram_range'])
    max_features = syntactic_config['max_features']
    min_df = syntactic_config['min_df']

    print(f"   spaCy model: {spacy_model_name}")
    print(f"   N-gram range: {ngram_range}")
    print(f"   Max features: {max_features}")
    print(f"   Min document frequency: {min_df}")

    # Load spaCy model
    print(f"   Loading spaCy model...")
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"   Model not found. Downloading {spacy_model_name}...")
        os.system(f"python -m spacy download {spacy_model_name}")
        nlp = spacy.load(spacy_model_name)

    # Disable unnecessary components for speed
    nlp.disable_pipes(['ner', 'parser'])

    # Extract POS sequences
    print(f"   Extracting POS tag sequences...")
    pos_sequences = extract_pos_tags(documents, nlp)

    # Create bag of POS n-grams
    print(f"   Creating bag of POS n-grams...")
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        token_pattern=r'\b\w+\b'  # Simple word pattern for POS tags
    )

    features = vectorizer.fit_transform(pos_sequences)

    print(f"   ✓ Extracted features: shape {features.shape}")
    print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")

    return features, vectorizer


def extract_morphological_features(documents, config):
    """Extract morphological features using POS sequences.

    Args:
        documents: List of document strings
        config: Configuration dict

    Returns:
        scipy sparse matrix of shape (n_documents, n_features)
    """
    print(f"\n4. Extracting morphological features...")

    morph_config = config['features']['morphological']
    spacy_model_name = morph_config['spacy_model']
    ngram_range = tuple(morph_config['ngram_range'])
    max_features = morph_config['max_features']
    min_df = morph_config['min_df']

    print(f"   N-gram range: {ngram_range}")
    print(f"   Max features: {max_features}")

    # Load spaCy model (reuse if already loaded)
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"   Downloading {spacy_model_name}...")
        os.system(f"python -m spacy download {spacy_model_name}")
        nlp = spacy.load(spacy_model_name)

    nlp.disable_pipes(['ner', 'parser'])

    # Extract POS sequences
    print(f"   Extracting POS sequences...")
    pos_sequences = []

    for doc in tqdm(documents, desc="   Processing"):
        try:
            parsed = nlp(doc[:5000])  # Limit for efficiency
            pos_tags = [token.pos_ for token in parsed if not token.is_space]
            pos_sequence = ' '.join(pos_tags)
            pos_sequences.append(pos_sequence)
        except:
            pos_sequences.append("")

    # Create bag of POS n-grams
    print(f"   Creating bag of POS n-grams...")
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        token_pattern=r'\b\w+\b'
    )

    features = vectorizer.fit_transform(pos_sequences)

    print(f"   ✓ Extracted features: shape {features.shape}")

    return features, vectorizer


def extract_phonological_features(documents, config):
    """Extract phonological features using phoneme sequences.

    Args:
        documents: List of document strings
        config: Configuration dict

    Returns:
        scipy sparse matrix of shape (n_documents, n_features)
    """
    print(f"\n5. Extracting phonological features...")

    phono_config = config['features']['phonological']
    backend = phono_config['backend']
    ngram_range = tuple(phono_config['ngram_range'])
    max_features = phono_config['max_features']
    min_df = phono_config['min_df']

    print(f"   Backend: {backend}")
    print(f"   N-gram range: {ngram_range}")

    # Import g2p_en for phoneme conversion
    try:
        from g2p_en import G2p
        g2p = G2p()
    except ImportError:
        print(f"   Installing g2p_en...")
        os.system("pip install g2p_en")
        from g2p_en import G2p
        g2p = G2p()

    # Extract phoneme sequences
    print(f"   Converting to phonemes...")
    phoneme_sequences = []

    for doc in tqdm(documents, desc="   Processing"):
        try:
            # Take first 200 words for efficiency
            words = doc.split()[:200]
            text = ' '.join(words)

            # Convert to phonemes
            phonemes = g2p(text)

            # Filter out empty phonemes and join
            phonemes = [p for p in phonemes if p and p.strip()]

            if len(phonemes) > 0:
                phoneme_sequence = ' '.join(phonemes)
                phoneme_sequences.append(phoneme_sequence)
            else:
                # Fallback: use words as phoneme-like features
                phoneme_sequences.append(' '.join(words[:50]))
        except Exception as e:
            # Fallback: use words
            words = doc.split()[:50]
            if words:
                phoneme_sequences.append(' '.join(words))
            else:
                phoneme_sequences.append("UNK")

    # Create bag of phoneme n-grams
    print(f"   Creating bag of phoneme n-grams...")
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=max(1, min_df),  # Ensure at least 1
        token_pattern=r'\S+',
        lowercase=True
    )

    try:
        features = vectorizer.fit_transform(phoneme_sequences)
        print(f"   ✓ Extracted features: shape {features.shape}")
    except ValueError as e:
        # If vocabulary is empty, create dummy features
        print(f"   ⚠ Warning: {e}")
        print(f"   ⚠ Creating fallback features based on character n-grams")

        # Fallback: use character n-grams
        vectorizer = CountVectorizer(
            ngram_range=(2, 3),
            max_features=max_features,
            min_df=1,
            analyzer='char_wb'
        )

        # Use original documents if phoneme sequences are problematic
        fallback_docs = [doc[:500] for doc in phoneme_sequences]
        features = vectorizer.fit_transform(fallback_docs)
        print(f"   ✓ Fallback features: shape {features.shape}")

    return features, vectorizer


def save_features(embeddings, syntactic_features, morphological_features, phonological_features,
                 syntactic_vectorizer, morphological_vectorizer, phonological_vectorizer,
                 features_dir, dataset_name):
    """Save extracted features to disk.

    Args:
        embeddings: Semantic embedding array
        syntactic_features: Syntactic feature sparse matrix
        morphological_features: Morphological feature sparse matrix
        phonological_features: Phonological feature sparse matrix
        syntactic_vectorizer: Fitted syntactic vectorizer
        morphological_vectorizer: Fitted morphological vectorizer
        phonological_vectorizer: Fitted phonological vectorizer
        features_dir: Output directory
        dataset_name: Name of dataset
    """
    print(f"\n6. Saving features...")

    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

    # Save semantic embeddings
    semantic_file = features_dir / f"{clean_name}_semantic_embeddings.npy"
    np.save(semantic_file, embeddings)
    print(f"   ✓ Saved semantic embeddings: {semantic_file}")

    # Save syntactic features
    syntactic_file = features_dir / f"{clean_name}_syntactic_features.npz"
    save_npz(syntactic_file, syntactic_features)
    print(f"   ✓ Saved syntactic features: {syntactic_file}")

    # Save morphological features
    morphological_file = features_dir / f"{clean_name}_morphological_features.npz"
    save_npz(morphological_file, morphological_features)
    print(f"   ✓ Saved morphological features: {morphological_file}")

    # Save phonological features
    phonological_file = features_dir / f"{clean_name}_phonological_features.npz"
    save_npz(phonological_file, phonological_features)
    print(f"   ✓ Saved phonological features: {phonological_file}")

    # Save vectorizers
    syntactic_vec_file = features_dir / f"{clean_name}_syntactic_vectorizer.pkl"
    with open(syntactic_vec_file, 'wb') as f:
        pickle.dump(syntactic_vectorizer, f)

    morphological_vec_file = features_dir / f"{clean_name}_morphological_vectorizer.pkl"
    with open(morphological_vec_file, 'wb') as f:
        pickle.dump(morphological_vectorizer, f)

    phonological_vec_file = features_dir / f"{clean_name}_phonological_vectorizer.pkl"
    with open(phonological_vec_file, 'wb') as f:
        pickle.dump(phonological_vectorizer, f)

    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'n_documents': embeddings.shape[0],
        'semantic_dim': embeddings.shape[1],
        'syntactic_vocab_size': syntactic_features.shape[1],
        'morphological_vocab_size': morphological_features.shape[1],
        'phonological_vocab_size': phonological_features.shape[1],
        'semantic_file': str(semantic_file),
        'syntactic_file': str(syntactic_file),
        'morphological_file': str(morphological_file),
        'phonological_file': str(phonological_file),
    }

    metadata_file = features_dir / f"{clean_name}_features_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✓ Saved metadata: {metadata_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("STEP 2: EXTRACT FEATURES")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Setup directories
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "input"
    features_dir = base_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']

        print(f"\n{'=' * 80}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 80}")

        try:
            # Load documents
            print(f"\n1. Loading documents...")
            documents = load_documents(input_dir, dataset_name)

            # Extract semantic features
            embeddings = extract_semantic_features(documents, config)

            # Extract syntactic features
            syntactic_features, syntactic_vectorizer = extract_syntactic_features(documents, config)

            # Extract morphological features
            morphological_features, morphological_vectorizer = extract_morphological_features(documents, config)

            # Extract phonological features
            phonological_features, phonological_vectorizer = extract_phonological_features(documents, config)

            # Save features
            save_features(
                embeddings,
                syntactic_features,
                morphological_features,
                phonological_features,
                syntactic_vectorizer,
                morphological_vectorizer,
                phonological_vectorizer,
                features_dir,
                dataset_name
            )

            print(f"\n{'=' * 80}")
            print(f"✓ Successfully processed: {dataset_name}")
            print(f"{'=' * 80}")

        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"✗ Failed to process: {dataset_name}")
            print(f"  Error: {e}")
            print(f"{'=' * 80}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✓ FEATURE EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
