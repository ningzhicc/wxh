"""Offline integration checks using a tiny, randomly initialized Llama, not the 1B model."""
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from kev import checkpoint, model
from kev.data import load_records, materialize
from runtime import CONTRACT_KEY, checkpoint_runtime, install_delimiters

ROOT = Path(__file__).resolve().parent


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.special = model.SPECIAL[:]
        self.loader = checkpoint.load_tokenizer

    def tearDown(self):
        model.SPECIAL[:] = self.special
        checkpoint.load_tokenizer = self.loader

    @staticmethod
    def tokenizer():
        words = ["[UNK]", "state", "buffer", "low", "high", "medium", "choose", "rate"]
        reserved = [f"<|reserved_special_token_{i}|>" for i in range(5)]
        vocab = {word: i for i, word in enumerate(words + reserved)}
        raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]",
                                      additional_special_tokens=reserved)

    def test_rejects_wrong_or_changed_delimiters(self):
        tok = self.tokenizer()
        contract = install_delimiters(tok)
        altered = {**contract, "ids": list(reversed(contract["ids"]))}
        with self.assertRaisesRegex(ValueError, "does not match"):
            install_delimiters(tok, altered)
        with self.assertRaisesRegex(ValueError, "Invalid Llama"):
            install_delimiters(tok, {"tokens": self.special, "ids": [1, 2, 3, 4, 5], "version": 1})

    def test_official_training_epoch_saves_and_reload(self):
        with tempfile.TemporaryDirectory(prefix="llama-abr-test-") as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            tok = self.tokenizer()
            tok.save_pretrained(base)
            tiny = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32,
                intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=2048))
            tiny.save_pretrained(base)
            data = root / "train.jsonl"
            record = {"state": "state buffer low", "questions": {"rate": {"type": "choice",
                "instructions": "choose rate", "criteria": {"low": "low", "medium": "medium", "high": "high"},
                "label": "medium"}}}
            data.write_text("\n".join(json.dumps(record) for _ in range(4)) + "\n")
            out, snapshots = root / "run", root / "checkpoints"
            command = [sys.executable, str(ROOT / "train.py"), "--epoch-checkpoints", str(snapshots),
                "--base", str(base), "--data", str(data), "--out", str(out), "--epochs", "2",
                "--lr", "2e-4", "--lora", "2", "--head_dim", "8", "--batch", "2", "--accum", "1",
                "--device", "cpu", "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0", "--seed", "666"]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])
            metrics = json.loads((out / "training_metrics.json").read_text())
            self.assertEqual(metrics["optimizer_steps"], 4)
            self.assertEqual(metrics["records_seen"], 8)
            self.assertEqual(metrics["rejected_records"], 0)
            for path in [out, snapshots / "epoch_01", snapshots / "epoch_02"]:
                meta = checkpoint_runtime(path)
                self.assertEqual(meta.base, str(base))
                self.assertEqual(len(meta.extra[CONTRACT_KEY]["ids"]), 5)
                loaded_tok, loaded = checkpoint.Checkpoint(path).load("cpu", checkpoint.LoadOptions(merge=False))
                encoded = loaded.encode(loaded_tok, materialize(load_records(data)[0]), strict=True)
                probability = loaded.probs(encoded)[0]
                self.assertTrue(torch.isfinite(probability).all())
                self.assertAlmostEqual(float(probability.sum()), 1.0, places=5)
                for tensor in loaded.head.state_dict().values():
                    self.assertTrue(torch.isfinite(tensor).all())
            first = checkpoint.read_meta(snapshots / "epoch_01")
            second = checkpoint.read_meta(snapshots / "epoch_02")
            self.assertTrue(any(not torch.equal(first.head[k], second.head[k]) for k in first.head))
            altered = dataclasses.replace(second, extra={})
            checkpoint.write_meta(snapshots / "epoch_02", altered)
            with self.assertRaisesRegex(ValueError, "Not a supported"):
                checkpoint_runtime(snapshots / "epoch_02")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
