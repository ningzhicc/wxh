"""Llama-only entry-point adaptations; leave the upstream Kev package untouched."""
import dataclasses
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache/huggingface"))

from transformers import AutoConfig, AutoTokenizer

from kev import checkpoint, model

CONTRACT_KEY = "llama_abr_encoding"
CONTRACT_VERSION = 1
RESERVED = re.compile(r"<\|reserved_special_token_\d+\|>")


def install_delimiters(tok, expected=None):
    """Reuse five existing reserved IDs; never grow or retrain the vocabulary."""
    vocab = tok.get_vocab()
    names = (expected["tokens"] if expected else
             sorted((t for t in vocab if RESERVED.fullmatch(t)), key=vocab.__getitem__)[:5])
    if len(names) != 5 or len(set(names)) != 5:
        raise ValueError("Llama encoding requires five distinct reserved special tokens")
    ids = []
    for name in names:
        if not RESERVED.fullmatch(name) or name not in vocab:
            raise ValueError(f"Invalid Llama delimiter: {name}")
        token_id = tok.convert_tokens_to_ids(name)
        if tok.encode(name, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Delimiter must encode as one token: {name}")
        ids.append(token_id)
    if len(set(ids)) != 5:
        raise ValueError("Delimiter IDs are not distinct")
    contract = {"version": CONTRACT_VERSION, "tokens": list(names), "ids": ids}
    if expected is not None and contract != expected:
        raise ValueError("Saved Llama delimiter contract does not match the tokenizer")
    model.SPECIAL[:] = names
    return contract


def load_llama_tokenizer(name, revision=None, *, expected=None, local_files_only=False):
    cfg = AutoConfig.from_pretrained(name, revision=revision, local_files_only=local_files_only)
    if cfg.model_type != "llama":
        raise ValueError(f"This entry point requires a Llama backbone, got {cfg.model_type}")
    tok = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local_files_only)
    install_delimiters(tok, expected)
    return tok


def training_runtime():
    """Call before official train.main(), in a dedicated Llama process."""
    from kev import train
    train.load_tokenizer = load_llama_tokenizer


def checkpoint_meta(meta, tok):
    return dataclasses.replace(meta, extra={**meta.extra, CONTRACT_KEY: install_delimiters(tok)})


def checkpoint_runtime(path):
    """Fail closed on a Qwen checkpoint or one lacking our saved encoding contract."""
    path = Path(path).resolve()
    meta = checkpoint.read_meta(path)
    expected = meta.extra.get(CONTRACT_KEY)
    if not isinstance(expected, dict) or expected.get("version") != CONTRACT_VERSION:
        raise ValueError("Not a supported Llama ABR checkpoint; use the original Kev entry point for Qwen")

    def loader(name, revision=None):
        return load_llama_tokenizer(name, revision, expected=expected)

    checkpoint.load_tokenizer = loader
    return meta
