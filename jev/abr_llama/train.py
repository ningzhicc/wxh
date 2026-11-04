"""Train a fresh Llama LoRA and pointer head using the existing Kev trainer."""
import argparse
import dataclasses
import math
import sys
from pathlib import Path

import torch

from runtime import checkpoint_meta, training_runtime
from kev import train
from kev.suite import digest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--epoch-checkpoints", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    if any(arg == "--init_from" or arg.startswith("--init_from=") for arg in remaining):
        raise SystemExit("This experiment starts fresh Llama adapters/head; do not pass a Kev/Qwen checkpoint")
    sys.argv = [sys.argv[0], *remaining]
    training_runtime()
    captured = {}
    original_model, original_requests, original_meta = train.DecisionModel, train.training_requests, train.Meta
    original_finish = train.finish_checkpoint
    original_scheduler = torch.optim.lr_scheduler.OneCycleLR

    def build_model(*values, **kwargs):
        result = original_model(*values, **kwargs)
        captured.update(model=result, tokenizer=values[1])
        return result

    def requests(*values, **kwargs):
        result = original_requests(*values, **kwargs)
        settings = values[0]
        if settings.full_ft or settings.row_budget or settings.pass_tokens_max:
            raise ValueError("This Llama experiment wrapper supports standard single-process LoRA batches only")
        if settings.data and len(result) != len(train.load_records(settings.data)):
            raise ValueError("Tokenizer filtering changed the training population; run prepare.py and resolve context overflow")
        captured.update(settings=values[0], count=len(result))
        return result

    def meta_factory(*values, **kwargs):
        result = original_meta(*values, **kwargs)
        captured["meta"] = result
        return result

    def finish(out, meta, tok):
        original_finish(out, checkpoint_meta(meta, tok), tok)

    class EpochScheduler(original_scheduler):
        def step(self, *values, **kwargs):
            super().step(*values, **kwargs)
            settings = captured["settings"]
            per_epoch = math.ceil(captured["count"] / (settings.batch * settings.accum))
            if self.last_epoch <= 0 or self.last_epoch % per_epoch:
                return
            epoch = self.last_epoch // per_epoch
            target = args.epoch_checkpoints / f"epoch_{epoch:02d}"
            if target.exists():
                raise ValueError(f"Refusing to overwrite {target}")
            pending = target.with_name(target.name + ".pending")
            pending.mkdir(parents=True, exist_ok=False)
            current = captured["model"]
            current.lm.save_pretrained(pending)
            meta = dataclasses.replace(captured["meta"],
                head={k: v.detach().cpu().clone() for k, v in current.head.state_dict().items()},
                extra={"args": vars(settings), "init_source": None,
                       "data_sha256": digest(settings.data) if settings.data else None,
                       "snapshot": {"epoch": epoch, "optimizer_steps": self.last_epoch}})
            finish(pending, meta, captured["tokenizer"])
            pending.rename(target)
            print(f"Saved Llama epoch {epoch}: {target}", flush=True)

    train.DecisionModel, train.training_requests, train.Meta = build_model, requests, meta_factory
    train.finish_checkpoint = finish
    torch.optim.lr_scheduler.OneCycleLR = EpochScheduler
    train.main()


if __name__ == "__main__":
    main()
