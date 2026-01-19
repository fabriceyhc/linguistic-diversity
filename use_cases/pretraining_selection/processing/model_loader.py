#!/usr/bin/env python3
"""Model loader with factory pattern for encoder, decoder, and encoder-decoder architectures.

Supports:
- Encoder-only: ModernBERT (answerdotai/ModernBERT-base)
- Decoder-only: Llama 3.2 1B (meta-llama/Llama-3.2-1B)
- Encoder-decoder: Flan-T5 Large (google/flan-t5-large)
"""

import os
from typing import Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

import torch
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@dataclass
class ModelConfig:
    """Configuration for a model type."""
    model_id: str
    model_class: type
    task_type: str
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = False
    max_length: int = 512
    extra_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}


# Model registry with configurations
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "encoder": ModelConfig(
        model_id="answerdotai/ModernBERT-base",
        model_class=AutoModelForMaskedLM,
        task_type="masked_lm",
        trust_remote_code=True,
        max_length=8192,  # 8k context window
        extra_kwargs={"attn_implementation": "eager"},  # For compatibility
    ),
    "decoder": ModelConfig(
        model_id="meta-llama/Llama-3.2-1B",
        model_class=AutoModelForCausalLM,
        task_type="causal_lm",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        max_length=2048,
    ),
    "encoder-decoder": ModelConfig(
        model_id="google/flan-t5-large",
        model_class=AutoModelForSeq2SeqLM,
        task_type="seq2seq_lm",
        trust_remote_code=False,
        max_length=512,
    ),
}


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: PreTrainedModel) -> Tuple[int, int]:
    """Count total and trainable parameters.

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_params(num: int) -> str:
    """Format parameter count in human-readable form."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def load_model(
    model_type: str,
    device: Optional[torch.device] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, ModelConfig]:
    """Load a model and tokenizer based on model type.

    Args:
        model_type: One of 'encoder', 'decoder', 'encoder-decoder'
        device: Device to load the model on. If None, auto-detects.
        load_in_8bit: Whether to load in 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Whether to load in 4-bit quantization (requires bitsandbytes)

    Returns:
        Tuple of (tokenizer, model, config)

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {list(MODEL_REGISTRY.keys())}"
        )

    config = MODEL_REGISTRY[model_type]
    device = device or get_device()

    print(f"{'=' * 60}")
    print(f"Loading {model_type} model: {config.model_id}")
    print(f"{'=' * 60}")
    print(f"Task type: {config.task_type}")
    print(f"Device: {device}")
    print(f"Max length: {config.max_length}")

    # Prepare model loading kwargs
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        **config.extra_kwargs,
    }

    # Handle dtype (use 'dtype' instead of deprecated 'torch_dtype')
    if config.torch_dtype is not None:
        model_kwargs["dtype"] = config.torch_dtype
        print(f"Dtype: {config.torch_dtype}")

    # Handle quantization
    if load_in_8bit or load_in_4bit:
        model_kwargs["device_map"] = "auto"
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            print("Loading in 8-bit quantization")
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            print("Loading in 4-bit quantization")

    # Check for HF token (needed for Llama)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        model_kwargs["token"] = hf_token
        tokenizer_kwargs = {"token": hf_token}
    else:
        tokenizer_kwargs = {}
        if model_type == "decoder":
            print("\nWARNING: No HF_TOKEN found. Llama 3.2 requires authentication.")
            print("Set HF_TOKEN environment variable or login with `huggingface-cli login`")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        **tokenizer_kwargs,
    )

    # Handle padding token for decoder models (Llama doesn't have one by default)
    if model_type == "decoder" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token: '{tokenizer.pad_token}'")

    # Load model
    print(f"Loading model...")
    try:
        model = config.model_class.from_pretrained(
            config.model_id,
            **model_kwargs,
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        if "token" in str(e).lower() or "auth" in str(e).lower():
            print("\nThis model may require authentication.")
            print("Please run: huggingface-cli login")
            print("Or set the HF_TOKEN environment variable.")
        raise

    # Move to device if not using device_map
    if "device_map" not in model_kwargs:
        model = model.to(device)

    # Count and print parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nParameter count:")
    print(f"  Total parameters:     {format_params(total_params)} ({total_params:,})")
    print(f"  Trainable parameters: {format_params(trainable_params)} ({trainable_params:,})")
    print(f"{'=' * 60}\n")

    return tokenizer, model, config


class TokenizationWrapper:
    """Standardized tokenization wrapper for different architectures."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_type: str,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ):
        """Initialize the tokenization wrapper.

        Args:
            tokenizer: The tokenizer to use
            model_type: One of 'encoder', 'decoder', 'encoder-decoder'
            max_length: Maximum sequence length
            device: Device to put tensors on
        """
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length
        self.device = device or get_device()

    def __call__(
        self,
        text: Union[str, list],
        target_text: Optional[Union[str, list]] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text for the specific architecture.

        Args:
            text: Input text or list of texts
            target_text: Target text for encoder-decoder models (optional)
            return_tensors: Format for return tensors

        Returns:
            Dictionary of tokenized inputs ready for the model
        """
        if self.model_type == "encoder":
            return self._tokenize_encoder(text, return_tensors)
        elif self.model_type == "decoder":
            return self._tokenize_decoder(text, return_tensors)
        elif self.model_type == "encoder-decoder":
            return self._tokenize_encoder_decoder(text, target_text, return_tensors)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _tokenize_encoder(
        self,
        text: Union[str, list],
        return_tensors: str,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize for encoder-only models (BERT-style)."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _tokenize_decoder(
        self,
        text: Union[str, list],
        return_tensors: str,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize for decoder-only models (GPT/Llama-style)."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        # For causal LM, labels are the same as input_ids (shifted internally)
        inputs["labels"] = inputs["input_ids"].clone()
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _tokenize_encoder_decoder(
        self,
        text: Union[str, list],
        target_text: Optional[Union[str, list]],
        return_tensors: str,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize for encoder-decoder models (T5-style)."""
        # If target text is provided, tokenize both together using text_target
        if target_text is not None:
            inputs = self.tokenizer(
                text,
                text_target=target_text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors=return_tensors,
            )
            # decoder_input_ids are created automatically from labels during forward pass
        else:
            # For inference without targets, tokenize input only
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors=return_tensors,
            )
            # Create dummy decoder_input_ids for inference
            # T5 uses pad_token_id as the start token for generation
            decoder_start_token_id = self.tokenizer.pad_token_id
            batch_size = inputs["input_ids"].shape[0]
            inputs["decoder_input_ids"] = torch.full(
                (batch_size, 1),
                decoder_start_token_id,
                dtype=torch.long,
            )

        return {k: v.to(self.device) for k, v in inputs.items()}


def smoke_test(model_type: str) -> Dict[str, Any]:
    """Run a smoke test for the specified model type.

    Args:
        model_type: One of 'encoder', 'decoder', 'encoder-decoder'

    Returns:
        Dictionary with test results
    """
    print(f"\n{'#' * 60}")
    print(f"SMOKE TEST: {model_type}")
    print(f"{'#' * 60}\n")

    # Load model
    tokenizer, model, config = load_model(model_type)
    device = next(model.parameters()).device

    # Create tokenization wrapper
    wrapper = TokenizationWrapper(
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=config.max_length,
        device=device,
    )

    # Test sentence
    test_sentence = "The quick brown fox jumps over the lazy dog."
    print(f"Test sentence: '{test_sentence}'")

    # Tokenize
    if model_type == "encoder-decoder":
        # For T5, we need a task prefix and target
        inputs = wrapper(
            f"summarize: {test_sentence}",
            target_text="A fox jumps over a dog.",
        )
    else:
        inputs = wrapper(test_sentence)

    print(f"\nTokenized inputs:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # Forward pass
    print(f"\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Report output shapes
    print(f"\nOutput shapes:")
    results = {"model_type": model_type, "model_id": config.model_id}

    if hasattr(outputs, "logits"):
        print(f"  logits: {outputs.logits.shape}")
        results["logits_shape"] = list(outputs.logits.shape)

    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        print(f"  hidden_states: {len(outputs.hidden_states)} layers")
        print(f"    last layer: {outputs.hidden_states[-1].shape}")
        results["hidden_states_layers"] = len(outputs.hidden_states)
        results["last_hidden_shape"] = list(outputs.hidden_states[-1].shape)

    if hasattr(outputs, "last_hidden_state"):
        print(f"  last_hidden_state: {outputs.last_hidden_state.shape}")
        results["last_hidden_state_shape"] = list(outputs.last_hidden_state.shape)

    if hasattr(outputs, "loss") and outputs.loss is not None:
        print(f"  loss: {outputs.loss.item():.4f}")
        results["loss"] = outputs.loss.item()

    if hasattr(outputs, "encoder_last_hidden_state"):
        print(f"  encoder_last_hidden_state: {outputs.encoder_last_hidden_state.shape}")
        results["encoder_last_hidden_state_shape"] = list(outputs.encoder_last_hidden_state.shape)

    print(f"\n{'#' * 60}")
    print(f"SMOKE TEST PASSED: {model_type}")
    print(f"{'#' * 60}\n")

    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_all_smoke_tests() -> Dict[str, Dict[str, Any]]:
    """Run smoke tests for all model types.

    Returns:
        Dictionary mapping model_type to test results
    """
    results = {}

    for model_type in MODEL_REGISTRY.keys():
        try:
            results[model_type] = smoke_test(model_type)
        except Exception as e:
            print(f"\nFailed smoke test for {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    # Summary
    print(f"\n{'=' * 60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'=' * 60}")
    for model_type, result in results.items():
        if "error" in result:
            print(f"  {model_type}: FAILED - {result['error']}")
        else:
            print(f"  {model_type}: PASSED ({result['model_id']})")
    print(f"{'=' * 60}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and test models")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["encoder", "decoder", "encoder-decoder", "all"],
        default="all",
        help="Model type to test",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )

    args = parser.parse_args()

    if args.model_type == "all":
        run_all_smoke_tests()
    else:
        smoke_test(args.model_type)
