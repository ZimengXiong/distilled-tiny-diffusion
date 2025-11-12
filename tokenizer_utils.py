"""Utility helpers for managing the project tokenizer.

The tokenizer is a simple BPE model trained on the Tiny Shakespeare
corpus. The trained tokenizer is cached on disk so training scripts and
inference utilities share the same vocabulary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Default locations and configuration
_TOKENIZER_CACHE = Path("weights/tokenizer.json")
_DEFAULT_DATASET = Path("data/tiny_shakespeare.txt")
_DEFAULT_VOCAB_SIZE = 4096
_SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]", "[BOS]", "[EOS]"]

# The cache keeps a singleton tokenizer instance per process
_TOKENIZER_INSTANCE: Optional[Tokenizer] = None

def _ensure_parent(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def _train_tokenizer(
    *,
    dataset_paths: Iterable[Path],
    vocab_size: int,
    save_path: Path,
) -> Tokenizer:
    """Train a new BPE tokenizer and persist it to save_path."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=_SPECIAL_TOKENS,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
    )
    print(f"Training tokenizer on {[str(p) for p in dataset_paths]}...")
    tokenizer.train([str(p) for p in dataset_paths], trainer)
    _ensure_parent(save_path)
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to {save_path}")
    return tokenizer

def get_tokenizer(
    *,
    data_paths: Sequence[Path] | None = None,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
    cache_path: Path | None = None,
) -> Tokenizer:
    """Return a shared tokenizer instance, training it if needed."""
    global _TOKENIZER_INSTANCE

    if _TOKENIZER_INSTANCE is not None:
        return _TOKENIZER_INSTANCE

    cache = cache_path or _TOKENIZER_CACHE
    if cache.exists():
        print(f"Loading tokenizer from {cache}")
        _TOKENIZER_INSTANCE = Tokenizer.from_file(str(cache))
        assert _TOKENIZER_INSTANCE is not None
        return _TOKENIZER_INSTANCE

    dataset_paths = list(data_paths) if data_paths is not None else [_DEFAULT_DATASET]
    _TOKENIZER_INSTANCE = _train_tokenizer(
        dataset_paths=dataset_paths,
        vocab_size=vocab_size,
        save_path=cache,
    )
    assert _TOKENIZER_INSTANCE is not None
    return _TOKENIZER_INSTANCE

def encode_text(text: str, *, add_bos: bool = False, add_eos: bool = False) -> torch.Tensor:
    """Encode text into a tensor of token ids using the shared tokenizer."""
    tokenizer = get_tokenizer()
    ids: List[int] = tokenizer.encode(text).ids

    if add_bos:
        bos_id = tokenizer.token_to_id("[BOS]")
        if bos_id is not None:
            ids.insert(0, bos_id)
    if add_eos:
        eos_id = tokenizer.token_to_id("[EOS]")
        if eos_id is not None:
            ids.append(eos_id)

    return torch.tensor(ids, dtype=torch.long)

def decode_tokens(tokens: Sequence[int] | torch.Tensor) -> str:
    """Decode a sequence of token ids back to text."""
    tokenizer = get_tokenizer()

    if isinstance(tokens, torch.Tensor):
        flat_tokens = tokens.detach().cpu().long().tolist()
    else:
        flat_tokens = [int(t) for t in tokens]

    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is not None:
        flat_tokens = [t for t in flat_tokens if t != pad_id]

    if not flat_tokens:
        return ""

    text = tokenizer.decode(flat_tokens, skip_special_tokens=False)
    for token in ("[BOS]", "[EOS]", "[PAD]"):
        text = text.replace(token, "")
    return text

def token_to_piece(token_id: int) -> str:
    """Return the raw text piece associated with a single token id."""
    tokenizer = get_tokenizer()
    piece = tokenizer.id_to_token(token_id)
    return piece if piece is not None else ""

def mask_token_id() -> int:
    """Convenience accessor for the tokenizer's mask token id."""
    tokenizer = get_tokenizer()
    mask_id = tokenizer.token_to_id("[MASK]")
    if mask_id is None:
        raise ValueError("Tokenizer is missing the [MASK] token.")
    return mask_id

def vocab_size() -> int:
    """Return the tokenizer vocabulary size."""
    tokenizer = get_tokenizer()
    return tokenizer.get_vocab_size()
