"""User-launched, resumable Kev LoRA training with two independent ABR modules."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

from checkpoints import latest_resume, prune_resume, publish
from io_utils import DATA, JEV, ROOT, atomic_json, digest, fingerprint, verify_data
from losses import decision_loss

VARIANTS = {(0, 0): "baseline", (1, 0): "cost", (0, 1): "regret", (1, 1): "both"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cost", type=int, choices=[0, 1], default=0)
    parser.add_argument("--regret-loss", type=int, choices=[0, 1], default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--regret-weight", type=float, default=.2)
    parser.add_argument("--regret-scale", type=float, default=4.3)
    parser.add_argument("--regret-clip", type=float, default=20)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--max-state", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--stop-after-steps", type=int, default=0, help="Manual stop at an optimizer boundary; schedule remains full length")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check configuration only: no model load, GPU work, run creation or training")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch, args.accum, args.save_every, args.log_every) < 1 or args.stop_after_steps < 0:
        parser.error("Epochs, batch, accumulation and intervals must be positive")
    if not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.regret_weight, args.regret_scale, args.regret_clip)):
        parser.error("Hyperparameters must be finite")
    if args.lr <= 0 or min(args.weight_decay, args.regret_weight) < 0 or min(args.regret_scale, args.regret_clip) <= 0:
        parser.error("Invalid optimizer / regret settings")
    args.variant = VARIANTS[(args.candidate_cost, args.regret_loss)]
    args.out = (args.out or ROOT / "runs" / f"merina656_{args.variant}_seed{args.seed}").resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def make_declaration(args):
    manifest = verify_data(args.data_dir)
    audit_path = args.data_dir / "preflight.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["complete"] or audit["data_fingerprint"] != manifest["fingerprint"] or audit["counts"] != manifest["counts"]:
        raise ValueError("Full paired preflight is missing or belongs to different data")
    if audit["limits"]["max_state"] != args.max_state or audit["dropped"] or audit["truncated"]:
        raise ValueError("Training context differs from full data audit")
    deployment = json.loads((JEV / "deployment.json").read_text(encoding="utf-8"))
    init = Path(deployment["checkpoint_path"])
    if digest(init / "adapter_model.safetensors") != deployment["adapter_sha256"]:
        raise ValueError("Released initialization changed")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}
    paths = [ROOT / n for n in ("train.py", "losses.py", "checkpoints.py", "io_utils.py")]
    paths += [JEV / "kev/kev" / n for n in ("model.py", "checkpoint.py", "data.py", "api.py", "device.py")]
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("resume", "dry_run", "stop_after_steps", "save_every", "log_every")}
    group = args.batch * args.accum
    declaration = {"settings": settings, "data_fingerprint": manifest["fingerprint"],
        "data_file": str(args.data_dir / ("cost.jsonl" if args.candidate_cost else "base.jsonl")),
        "data_sha256": manifest["files_sha256"]["cost.jsonl" if args.candidate_cost else "base.jsonl"],
        "preflight_fingerprint": fingerprint({k: v for k, v in audit.items() if k != "seconds"}),
        "records_per_epoch": manifest["counts"]["records"],
        "regret_eligible_records": manifest["counts"]["regret_valid"], "effective_batch": group,
        "steps_per_epoch": math.ceil(manifest["counts"]["records"] / group),
        "optimizer_steps": args.epochs * math.ceil(manifest["counts"]["records"] / group),
        "requested_records": args.epochs * manifest["counts"]["records"],
        "base": deployment["base"], "base_revision": deployment["base_revision"],
        "base_local_path": deployment["base_path"], "init_checkpoint": str(init),
        "init_sha256": {n: digest(init / n) for n in ("adapter_model.safetensors", "adapter_config.json", "head.pt")},
        "architecture": {"lora": 16, "lora_targets": "all", "head_dim": 256, "weights_dtype": "fp32", "dtype": "bf16"},
        "objective": "hard_action_CE + enabled * weight * expected_clipped_one_step_proxy_regret / scale",
        "validation": "Save every epoch; select checkpoints later using the same FCC validation protocol. No test-based selection.",
        "versions": versions, "python": sys.version.split()[0],
        "runtime_code_sha256": {str(p.relative_to(JEV)): digest(p) for p in paths}}
    declaration["fingerprint"] = fingerprint(declaration)
    return declaration


def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(f"abr_merina656:{seed}:{epoch}").shuffle(order)
    return order


@contextlib.contextmanager
def exclusive_training(run):
    # Only this package's jobs are locked. No other service/process is stopped.
    with (ROOT / ".training.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another abr_modular training job is running; run the four arms sequentially")
        yield


def train(args, declaration):
    os.environ.setdefault("HF_HOME", str(JEV / ".cache/huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(JEV / "kev"))
    import torch
    from transformers import AutoTokenizer
    from kev.checkpoint import Checkpoint, Meta
    from kev.data import load_records, materialize
    from kev.model import DecisionModel, training_context

    run = args.out
    if run.exists() and not args.resume:
        raise ValueError("Run exists; use --resume or a new --out")
    if args.resume and not run.exists():
        raise ValueError("Resume run does not exist")
    with exclusive_training(run):
        run.mkdir(parents=True, exist_ok=args.resume)
        config_path = run / "experiment.json"
        if args.resume:
            old = json.loads(config_path.read_text(encoding="utf-8"))
            if old["fingerprint"] != declaration["fingerprint"]:
                raise ValueError("Resume configuration, data, initialization, environment or code changed")
            if (run / "training_metrics.json").is_file():
                print("Run already complete:", run, flush=True)
                return
            complete_points = list((run / "resume").glob("step_*/complete.json"))
            resume_path = latest_resume(run, declaration["fingerprint"]) if complete_points else None
            if resume_path is None:
                print("No complete optimizer point exists yet; restarting from the identical released initialization.", flush=True)
        else:
            atomic_json(config_path, declaration)
            resume_path = None
        if not torch.cuda.is_available():
            raise ValueError("CUDA training requires the existing Kev CUDA environment")
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tokenizer = AutoTokenizer.from_pretrained(declaration["base_local_path"], local_files_only=True)
        model = DecisionModel(declaration["base"], tokenizer, "cuda", lora=16,
            revision=declaration["base_revision"], head_dim=256, lora_targets="all", dtype=torch.float32)
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.lm.config.use_cache = False
        meta = Meta(base=declaration["base"], base_revision=declaration["base_revision"], lora=16,
                    head_dim=256, weights_dtype="fp32", temperature=1.0,
                    extra={"abr_modular": declaration, "candidate_cost": bool(args.candidate_cost), "regret_loss": bool(args.regret_loss)})
        initialization = Checkpoint(str(resume_path or declaration["init_checkpoint"])).warm_start(model, meta)
        requests = load_records(declaration["data_file"], source="abr_merina656")
        if len(requests) != declaration["records_per_epoch"]:
            raise ValueError("Training population changed")
        ids = [r["_meta"]["id"] for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("Training record IDs are not unique")
        params = model.trainable_parameters()
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                    total_steps=declaration["optimizer_steps"], pct_start=.1)
        position = {"epoch": 0, "next_group": 0, "step": 0, "records_seen": 0}
        metrics = {"ce_sum": 0.0, "regret_sum": 0.0, "regret_valid": 0, "examples": 0}
        previous_seconds = 0.0
        if resume_path:
            saved = torch.load(resume_path / "training_state.pt", map_location="cpu", weights_only=True)
            if saved["fingerprint"] != declaration["fingerprint"]:
                raise ValueError("Resume state mismatch")
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            position, metrics = saved["position"], saved["metrics"]
            previous_seconds = saved["train_seconds"]
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            print(f"Resumed step {position['step']}, next epoch index {position['epoch']}, group {position['next_group']}", flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        limits = training_context(args.max_state)
        started = time.monotonic()
        stop = {"requested": False}
        handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *unused: stop.update(requested=True))

        def save_resume():
            state = {"fingerprint": declaration["fingerprint"], "position": dict(position), "metrics": dict(metrics),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                     "train_seconds": previous_seconds + time.monotonic() - started}
            info = {"fingerprint": declaration["fingerprint"], **position}
            path = publish(run / "resume", f"step_{position['step']:08d}", model, tokenizer, meta, info, state)
            atomic_json(run / "last_resume.json", {"path": str(path), **position})
            prune_resume(run, declaration["fingerprint"])
            return path

        try:
            if resume_path is None:
                save_resume()
            group_size = declaration["effective_batch"]
            log_path = run / "steps.jsonl"
            for epoch in range(position["epoch"], args.epochs):
                order = epoch_order(len(requests), args.seed, epoch)
                first_group = position["next_group"] if epoch == position["epoch"] else 0
                for group in range(first_group, declaration["steps_per_epoch"]):
                    selected = order[group * group_size:(group + 1) * group_size]
                    terms = Counter()
                    for offset in range(0, len(selected), args.batch):
                        rows = [requests[i] for i in selected[offset:offset + args.batch]]
                        records = [materialize(r) for r in rows]
                        encoded = [model.encode(tokenizer, r, max_state=limits["max_state"], max_branch=limits["max_branch"], strict=True) for r in records]
                        if any(len(e["ids"]) > limits["max_packed"] for e in encoded):
                            raise ValueError("Context overflow; no samples may be dropped")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model.forward_batch(encoded)
                        losses = []
                        for row, rec, questions in zip(rows, records, logits, strict=True):
                            if len(questions) != 1 or len(rec["questions"]) != 1:
                                raise ValueError("Expected exactly one six-way ABR question")
                            q = rec["questions"][0]
                            loss, values = decision_loss(questions[0], q["label"], row["_supervision"],
                                enabled=bool(args.regret_loss), weight=args.regret_weight,
                                scale=args.regret_scale, clip=args.regret_clip, keys=q["keys"])
                            losses.append(loss)
                            terms.update(values)
                        (sum(losses) / len(selected)).backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    epoch_done = group + 1 == declaration["steps_per_epoch"]
                    position.update(epoch=epoch + 1 if epoch_done else epoch, next_group=0 if epoch_done else group + 1,
                                    step=position["step"] + 1, records_seen=position["records_seen"] + len(selected))
                    metrics["ce_sum"] += terms["ce"]
                    metrics["regret_sum"] += terms["regret"]
                    metrics["regret_valid"] += terms["regret_valid"]
                    metrics["examples"] += len(selected)
                    if position["step"] % args.log_every == 0 or epoch_done:
                        progress = {**position, "total_steps": declaration["optimizer_steps"], "variant": args.variant,
                                    "ce": terms["ce"] / len(selected), "regret": terms["regret"] / len(selected),
                                    "lr": scheduler.get_last_lr()[0], "grad_norm": norm,
                                    "train_seconds": previous_seconds + time.monotonic() - started}
                        with log_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(progress, allow_nan=False) + "\n")
                        atomic_json(run / "progress.json", progress)
                        print(f"{args.variant} ep {epoch + 1}/{args.epochs} step {position['step']}/{declaration['optimizer_steps']} CE={progress['ce']:.4f} regret={progress['regret']:.4f}", flush=True)
                    if epoch_done:
                        snapshot = publish(run / "checkpoints", f"epoch_{epoch + 1:02d}", model, tokenizer, meta,
                                           {"fingerprint": declaration["fingerprint"], **position})
                        atomic_json(run / "last_checkpoint.json", {"path": str(snapshot), **position})
                    stopping = stop["requested"] or bool(args.stop_after_steps and position["step"] >= args.stop_after_steps)
                    if position["step"] % args.save_every == 0 or epoch_done or stopping:
                        save_resume()
                    if stopping:
                        atomic_json(run / "status.json", {"stage": "stopped_at_optimizer_boundary", **position})
                        print("Saved complete optimizer/RNG state. Continue with the same command plus --resume.", flush=True)
                        return
            if position["step"] != declaration["optimizer_steps"] or position["records_seen"] != declaration["requested_records"]:
                raise ValueError("Training exposure counts differ from declaration")
            final = run / "checkpoints" / f"epoch_{args.epochs:02d}"
            atomic_json(run / "training_metrics.json", {"complete": True, **position, **metrics,
                "data_fingerprint": declaration["data_fingerprint"], "run_fingerprint": declaration["fingerprint"],
                "last_checkpoint": str(final), "selected_best_checkpoint": None,
                "train_seconds": previous_seconds + time.monotonic() - started,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "initialization": initialization})
            atomic_json(run / "status.json", {"stage": "training_complete_validation_pending", **position})
            print("Training complete; select a checkpoint on FCC validation before the four test sets:", final, flush=True)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    args = parse_args()
    declaration = make_declaration(args)
    if args.dry_run:
        print(json.dumps({**declaration, "dry_run": True, "model_loaded": False, "training_started": False}, indent=2))
        return
    train(args, declaration)


if __name__ == "__main__":
    main()
