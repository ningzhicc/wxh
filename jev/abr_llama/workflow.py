"""Persistent Llama ABR experiment: untrained test, validation early stopping, final test."""
import argparse
import dataclasses
import fcntl
import gc
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "models/Llama-3.2-1B"
PYTHON = ROOT.parent / "kev/.venv/bin/python"
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import VIDEO_DIR, atomic_json, canonical_hash, digest
sys.path.insert(0, str(ROOT.parent / "abr_finetune"))
from continue_training import consider, optimizer_device, stopping


def evaluate(checkpoint, scope, out, limit=None):
    command = [str(PYTHON), str(ROOT / "evaluate.py"), "--checkpoint", str(checkpoint),
               "--scope", scope, "--out", str(out)]
    if limit:
        command += ["--trace-limit", str(limit)]
    subprocess.run(command, check=True)
    return json.loads((out / "summary.json").read_text())


def validation(checkpoint, out, options):
    summaries = [evaluate(checkpoint, "selection", out / f"repeat_{i:02d}", 2 if options.probe else None)[0]
                 for i in range(1, options.repeats + 1)]
    return {"checkpoint": str(checkpoint), "score": sum(s["qoe_mean"] for s in summaries)/len(summaries),
            "repeat_summaries": summaries}


def save_model(model, tok, meta, target, extra):
    from runtime import checkpoint_meta
    from kev import train
    if target.exists():
        raise ValueError(f"Refusing to overwrite {target}")
    pending = target.with_name(target.name + ".pending")
    pending.mkdir(exist_ok=False)
    model.lm.save_pretrained(pending)
    snapshot_meta = dataclasses.replace(meta,
        head={k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}, extra=extra)
    train.finish_checkpoint(pending, checkpoint_meta(snapshot_meta, tok), tok)
    return pending


def build(settings):
    from runtime import load_llama_tokenizer
    from kev.model import DecisionModel
    from kev.checkpoint import Meta
    tok = load_llama_tokenizer(settings.base, local_files_only=True)
    model = DecisionModel(settings.base, tok, "cuda", lora=settings.lora, head_dim=settings.head_dim,
                          lora_targets=settings.lora_targets)
    meta = Meta(base=settings.base, lora=settings.lora, head_dim=settings.head_dim)
    return tok, model, meta


def run(options):
    import torch
    from kev import train
    from kev.checkpoint import Checkpoint
    out = ROOT / "results" / options.run_name
    run_dir = ROOT / "runs" / options.run_name
    checkpoints = ROOT / "checkpoints" / options.run_name
    for p in [run_dir, checkpoints]:
        p.mkdir(parents=True, exist_ok=True)
    if options.resume:
        for pending in checkpoints.glob("epoch_*.pending"):
            pending.rename(pending.with_name(pending.name + f".abandoned-{time.time_ns()}"))
    data_dir = ROOT.parent / "abr_finetune/data"
    frozen = json.loads((data_dir / "manifest.json").read_text())
    assert all(digest(data_dir / name) == expected for name, expected in frozen["files_sha256"].items())
    receipt = json.loads((ROOT / "model_download.json").read_text())
    expected = next(f["sha256"] for f in receipt["files"] if f["path"] == "model.safetensors")
    assert digest(BASE / "model.safetensors") == expected
    old = json.loads((ROOT.parent / "abr_finetune/runs/sft_v3/training_config.json").read_text())["args"]
    settings = argparse.Namespace(**{**old, "base": str(BASE), "base_revision": "", "init_from": "",
        "out": str(run_dir), "data": str(data_dir / "train.jsonl"), "epochs": options.max_epochs,
        "lr": 2e-5, "head_lr": 1e-4, "max_steps": 0})
    definition = {"options": {k: v for k, v in vars(options).items() if k not in {"detach", "resume"}},
        "training_args": vars(settings), "base_weights_sha256": expected,
        "base_config_sha256": digest(BASE / "config.json"), "data_manifest_sha256": digest(data_dir / "manifest.json"),
        "video_sha256": {f.name: digest(f) for f in sorted(VIDEO_DIR.glob("video_size_*"))},
        "source_sha256": {str(p.relative_to(ROOT.parent)): digest(p) for p in [
            Path(__file__), ROOT / "evaluate.py", ROOT / "runtime.py", ROOT / "serve.py", ROOT / "EXPERIMENT_PROTOCOL.md",
            ROOT.parent / "abr_eval/benchmark.py", ROOT.parent / "abr_finetune/evaluate_model.py",
            ROOT.parent / "abr_finetune/continue_training.py", ROOT.parent / "kev/kev/train.py"]}}
    fingerprint = canonical_hash(definition)
    config_path = run_dir / "experiment.json"
    if config_path.exists():
        assert options.resume and json.loads(config_path.read_text())["fingerprint"] == fingerprint
    else:
        atomic_json(config_path, {**definition, "fingerprint": fingerprint,
            "environment": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "triton"]}})
    baseline = checkpoints / "epoch_00"
    if not baseline.exists():
        atomic_json(out / "status.json", {"stage": "initializing_random_pointer", "seed": settings.seed})
        torch.manual_seed(settings.seed)
        tok, model, meta = build(settings)
        pending = save_model(model, tok, meta, baseline, {"fingerprint": fingerprint, "epoch": 0,
            "initialization": "pretrained Llama; identity LoRA; randomly initialized pointer head", "seed": settings.seed})
        atomic_json(pending / "snapshot.json", {"complete": True, "epoch": 0})
        pending.rename(baseline)
        del model, tok, meta
        gc.collect()
        torch.cuda.empty_cache()
    if not options.probe:
        atomic_json(out / "status.json", {"stage": "untrained_test", "checkpoint": str(baseline)})
        evaluate(baseline, "test", out / "untrained_test")
    ledger_path = out / "controller.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    else:
        atomic_json(out / "status.json", {"stage": "baseline_validation"})
        initial = {**validation(baseline, out / "validation_epoch_00", options), "epoch": 0}
        ledger = {"best": initial, "meaningful_best_score": initial["score"], "candidates": [initial],
                  "bad_epochs": 0, "last_validated_epoch": 0}
        atomic_json(ledger_path, ledger)
    torch.manual_seed(settings.seed)
    sampler = random.Random(settings.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    completed, step, records_seen, seconds = 0, 0, 0, 0.0
    history, gradients = [], []
    committed = sorted(p for p in checkpoints.glob("epoch_*") if p.name[-2:].isdigit()
                       and (p / "training_state.pt").exists() and (p / "snapshot.json").exists())
    resume = None
    parent = baseline
    if options.resume and committed:
        parent = committed[-1]
        marker = json.loads((parent / "snapshot.json").read_text())
        assert marker["complete"] and digest(parent / "training_state.pt") == marker["training_state_sha256"]
        resume = torch.load(parent / "training_state.pt", map_location="cpu", weights_only=False)
        assert resume["fingerprint"] == fingerprint
    tok, model, meta = build(settings)
    provenance = Checkpoint(parent).warm_start(model, meta)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    requests = train.training_requests(settings, tok, None, [])
    assert len(requests) == 7224
    if options.probe:
        requests = requests[:16]
    by_id = {r["_meta"]["id"]: r for r in requests}
    assert len(by_id) == len(requests)
    per_epoch = sum(end for _, _, end in train.microbatch_plan(requests, settings, 1, 0))
    trainable = model.trainable_parameters()
    parameter_ids = [id(p) for p in trainable]
    head = list(model.head.parameters())
    head_ids = {id(p) for p in head}
    groups = [{"params": [p for p in trainable if id(p) not in head_ids], "lr": settings.lr},
              {"params": head, "lr": settings.head_lr}]
    optimizer = torch.optim.AdamW(groups, lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[settings.lr, settings.head_lr],
        total_steps=options.max_epochs*per_epoch, pct_start=.1)
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        optimizer_device(optimizer, "cuda")
        scheduler.load_state_dict(resume["scheduler"])
        completed, step, records_seen, seconds = [resume[k] for k in ["epoch", "step", "records_seen", "seconds"]]
        history, gradients = resume["loss_history"], resume["gradients"]
        requests = [by_id[k] for k in resume["order"]]
        sampler.setstate(resume["sampler_rng"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
        del resume
        print("Restored complete optimizer/random states at epoch", completed, flush=True)
    print(f"Llama training: {len(requests)} records, {per_epoch} steps/epoch, "
          f"{sum(p.numel() for p in trainable)/1e6:.2f}M trainable parameters", flush=True)

    def validate(epoch, path):
        optimizer.zero_grad(set_to_none=True)
        weight_copies = [p.detach().cpu().clone() for p in trainable] if options.probe else None
        moment_copies = [{k: v.detach().cpu().clone() for k, v in state.items() if isinstance(v, torch.Tensor)}
                         for state in optimizer.state.values()] if options.probe else None
        model.to("cpu")
        optimizer_device(optimizer, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        atomic_json(out / "status.json", {"stage": "validation", "epoch": epoch,
            "training_gpu_bytes_after_offload": torch.cuda.memory_allocated()})
        score = {**validation(path, out / f"validation_epoch_{epoch:02d}", options), "epoch": epoch}
        consider(ledger, score, options.min_delta)
        ledger["last_validated_epoch"] = epoch
        atomic_json(ledger_path, ledger)
        model.to("cuda")
        optimizer_device(optimizer, "cuda")
        assert [id(p) for p in model.trainable_parameters()] == parameter_ids
        if options.probe:
            assert all(torch.equal(p.detach().cpu(), old) for p, old in zip(trainable, weight_copies))
            for state, old in zip(optimizer.state.values(), moment_copies):
                assert all(torch.equal(state[k].cpu(), value) for k, value in old.items())
        model.train()
        print("Validation", json.dumps({"epoch": epoch, "score": score["score"],
            "best_epoch": ledger["best"]["epoch"], "bad_epochs": ledger["bad_epochs"]}), flush=True)
        return stopping(ledger, epoch, options.min_epochs, options.max_epochs, options.patience)

    reason = None
    if completed > ledger["last_validated_epoch"]:
        reason = validate(completed, parent)
    else:
        reason = stopping(ledger, completed, options.min_epochs, options.max_epochs, options.patience)
    for epoch in range(completed + 1, options.max_epochs + 1):
        if reason:
            break
        started = time.monotonic()
        atomic_json(out / "status.json", {"stage": "training", "epoch": epoch, "step": step,
            "steps_per_epoch": per_epoch, "best_epoch": ledger["best"]["epoch"]})
        model.train()
        sampler.shuffle(requests)
        terms, norms = Counter(), []
        for chunk, step_records, end in train.microbatch_plan(requests, settings, 1, 0):
            batch = train.encode_batch(model, tok, settings, chunk, epoch-1)
            variants = sum(v.share for v in batch)
            group_records = step_records*variants/len(chunk)
            for part in train.row_passes(batch, settings.row_budget, settings.shared_prefix):
                loss, values = train.batch_loss(model, settings, part, "cuda", {}, None,
                                              torch.autocast("cuda", dtype=torch.bfloat16))
                (loss/group_records).backward()
                terms += values
            terms["n"] += variants
            records_seen += round(variants)
            if end:
                norms.append(float(torch.nn.utils.clip_grad_norm_(trainable, train.MAX_GRAD_NORM, error_if_nonfinite=True)))
                optimizer.step()
                torch.cuda.synchronize()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or options.probe:
                    row = {"epoch": epoch, "step": step, "loss": terms["ce"]/terms["n"], "records_seen": records_seen}
                    history.append(row)
                    atomic_json(out / "training_progress.json", row)
                    print(f"Epoch {epoch} step {step}/{options.max_epochs*per_epoch} CE {row['loss']:.4f}", flush=True)
                    terms = Counter()
        seconds += time.monotonic()-started
        assert step == epoch*per_epoch and records_seen == epoch*len(requests)
        gradients.append({"epoch": epoch, "steps": len(norms), "mean": sum(norms)/len(norms), "max": max(norms)})
        target = checkpoints / f"epoch_{epoch:02d}"
        pending = save_model(model, tok, meta, target, {"args": vars(settings), "fingerprint": fingerprint,
            "init_source": provenance, "snapshot": {"epoch": epoch, "optimizer_steps": step},
            "data_sha256": digest(settings.data)})
        state = {"fingerprint": fingerprint, "epoch": epoch, "step": step, "records_seen": records_seen,
            "seconds": seconds, "loss_history": history, "gradients": gradients,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "order": [r["_meta"]["id"] for r in requests], "sampler_rng": sampler.getstate(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        torch.save(state, pending / "training_state.pt")
        atomic_json(pending / "snapshot.json", {"complete": True, "epoch": epoch,
            "training_state_sha256": digest(pending / "training_state.pt")})
        pending.rename(target)
        del state, batch, loss
        completed = epoch
        print(f"Saved epoch {epoch}; cumulative optimization time {seconds/60:.2f} minutes", flush=True)
        reason = validate(epoch, target)
    selected = {**ledger["best"], "selected_epoch": ledger["best"]["epoch"], "last_trained_epoch": completed,
                "stop_reason": reason, "uses_test_for_selection": False, "candidates": ledger["candidates"]}
    atomic_json(out / "selected.json", selected)
    metrics = {"epochs_completed": completed, "optimizer_steps": step, "records_seen": records_seen,
        "records_per_epoch": len(requests), "steps_per_epoch": per_epoch, "optimization_seconds": seconds,
        "loss_history": history, "gradients": gradients, "peak_device_bytes": torch.cuda.max_memory_allocated(),
        "trainable_parameters": sum(p.numel() for p in trainable)}
    atomic_json(run_dir / "training_metrics.json", metrics)
    del model, trainable, head, groups, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    if options.probe:
        atomic_json(out / "completion.json", {"complete": True, "probe": True,
            "optimizer_weights_preserved_across_validation": True})
        atomic_json(out / "status.json", {"stage": "complete", "probe": True})
        return
    atomic_json(out / "status.json", {"stage": "final_test", "selected_epoch": selected["selected_epoch"]})
    if selected["selected_epoch"] == 0:
        (out / "test").symlink_to(out / "untrained_test", target_is_directory=True) if not (out / "test").exists() else None
    else:
        evaluate(Path(selected["checkpoint"]), "test", out / "test")
    subprocess.run([str(PYTHON), str(ROOT / "make_report.py"), "--run-name", options.run_name], check=True)
    atomic_json(out / "completion.json", {"complete": True, "selected_epoch": selected["selected_epoch"],
        "last_trained_epoch": completed, "stop_reason": reason, "test_episodes_before": 275, "test_episodes_after": 275})
    atomic_json(out / "status.json", {"stage": "complete", "selected_epoch": selected["selected_epoch"],
                                       "last_trained_epoch": completed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="llama_abr_v1")
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=.005)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    assert Path(options.run_name).name == options.run_name and options.run_name not in {".", ".."}
    assert 1 <= options.min_epochs <= options.max_epochs <= 20 and options.patience >= 1
    assert options.repeats > 0 and math.isfinite(options.min_delta) and options.min_delta > 0
    if options.probe:
        options.min_epochs = options.max_epochs = 2
        options.repeats = 1
    os.environ.update(HF_HOME=str(ROOT.parent / ".cache/huggingface"), HF_HUB_OFFLINE="1",
        XDG_CACHE_HOME=str(ROOT / ".cache"), TRITON_CACHE_DIR=str(ROOT / ".cache/triton"),
        PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    if options.detach:
        command = [str(PYTHON), str(Path(__file__).resolve()), "--run-name", options.run_name,
            "--min-epochs", str(options.min_epochs), "--max-epochs", str(options.max_epochs),
            "--patience", str(options.patience), "--min-delta", str(options.min_delta), "--repeats", str(options.repeats)]
        if options.probe:
            command.append("--probe")
        if options.resume:
            command.append("--resume")
        with (logs / f"{options.run_name}.log").open("a" if options.resume else "x") as stream:
            child = subprocess.Popen(command, env=os.environ, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (logs / f"{options.run_name}.pid").write_text(str(child.pid))
        print("Detached PID", child.pid, "log", logs / f"{options.run_name}.log", flush=True)
        return
    out = ROOT / "results" / options.run_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        print("Already complete:", out)
        return
    with (out / ".workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(options)
            (out / "failure.json").unlink(missing_ok=True)
        except BaseException as error:
            atomic_json(out / "failure.json", {"error": repr(error), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
