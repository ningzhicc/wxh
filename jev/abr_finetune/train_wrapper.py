"""Use the official Kev trainer and retain loadable LoRA checkpoints per epoch."""
import argparse
import dataclasses
import math
import sys
from pathlib import Path

import torch
from kev import train
from kev.checkpoint import Checkpoint
from kev.suite import digest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--epoch-checkpoints", required=True, type=Path)
wrapper_args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
captured = {}
original_model = train.DecisionModel
original_requests = train.training_requests
original_warm_start = Checkpoint.warm_start
original_scheduler = torch.optim.lr_scheduler.OneCycleLR


def build_model(*args, **kwargs):
    model = original_model(*args, **kwargs)
    captured["model"] = model
    captured["tokenizer"] = args[1]
    return model


def requests(*args, **kwargs):
    records = original_requests(*args, **kwargs)
    captured["args"] = args[0]
    captured["count"] = len(records)
    return records


def warm_start(self, model, meta):
    provenance = original_warm_start(self, model, meta)
    captured["meta"], captured["init_source"] = meta, provenance
    return provenance


class EpochScheduler(original_scheduler):
    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        settings = captured["args"]
        per_epoch = math.ceil(captured["count"] / (settings.batch * settings.accum))
        if self.last_epoch <= 0 or self.last_epoch % per_epoch:
            return
        epoch = self.last_epoch // per_epoch
        destination = wrapper_args.epoch_checkpoints / f"epoch_{epoch:02d}"
        if destination.exists():
            raise ValueError(f"Refusing to overwrite epoch checkpoint {destination}")
        destination.mkdir(parents=True)
        model = captured["model"]
        model.lm.save_pretrained(destination)
        meta = dataclasses.replace(captured["meta"],
            head={key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()},
            extra={"args": vars(settings), "init_source": captured["init_source"],
                   "data_sha256": digest(settings.data),
                   "snapshot": {"epoch": epoch, "optimizer_steps": self.last_epoch}})
        train.finish_checkpoint(destination, meta, captured["tokenizer"])
        print(f"Saved epoch {epoch} checkpoint: {destination}", flush=True)


train.DecisionModel = build_model
train.training_requests = requests
Checkpoint.warm_start = warm_start
torch.optim.lr_scheduler.OneCycleLR = EpochScheduler
train.main()
