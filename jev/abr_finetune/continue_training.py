"""Continue ABR LoRA training with persistent optimizer state and validation early stopping."""
import argparse
import copy
import dataclasses
import fcntl
import gc
import importlib.metadata
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STOCK = ROOT.parent / "kev/.venv/bin/python"
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import atomic_json, canonical_hash, digest


def consider(ledger, candidate, min_delta):
    """Track absolute best separately from improvements large enough to reset patience."""
    ledger["candidates"].append(candidate)
    if candidate["score"] > ledger["best"]["score"] + 1e-12:
        ledger["best"] = candidate
    if candidate["score"] > ledger["meaningful_best_score"] + min_delta:
        ledger["meaningful_best_score"] = candidate["score"]
        ledger["bad_epochs"] = 0
    else:
        ledger["bad_epochs"] += 1
    return ledger


def stopping(ledger, epoch, min_epochs, max_epochs, patience):
    if epoch >= min_epochs and ledger["bad_epochs"] >= patience:
        return "validation_plateau"
    if epoch >= max_epochs:
        return "maximum_epochs_reached"
    return None


def optimizer_device(optimizer, device):
    import torch
    # AdamW's non-capturable scalar step counters stay on CPU as in the official trainer.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)


def evaluate(checkpoint, out, repeats=2, trace_limit=None):
    rows = []
    for repeat in range(1, repeats + 1):
        destination = out / f"repeat_{repeat:02d}"
        command = [str(STOCK), str(ROOT / "evaluate_model.py"), "--checkpoint", str(checkpoint),
                   "--scope", "selection", "--out", str(destination)]
        if trace_limit:
            command += ["--trace-limit", str(trace_limit)]
        subprocess.run(command, check=True)
        rows.append(json.loads((destination / "summary.json").read_text(encoding="utf-8"))[0])
    result = {"score": sum(r["qoe_mean"] for r in rows) / len(rows), "repeat_summaries": rows,
              "checkpoint": str(checkpoint), "validation_episodes_per_repeat": rows[0]["episodes"]}
    atomic_json(out / "score.json", result)
    return result


def run(options):
    import torch
    from kev import train
    from kev.checkpoint import Checkpoint, Meta
    from kev.model import DecisionModel, load_tokenizer
    from kev.data import load_records
    run_name = options.run_name
    out = ROOT / "results" / run_name
    run_dir = ROOT / "runs" / run_name
    checkpoints = ROOT / "checkpoints" / run_name
    out.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    assert all(digest(ROOT / "data" / name) == sha for name, sha in frozen["files_sha256"].items())
    source_settings = json.loads((ROOT / "runs/sft_v3/training_config.json").read_text(encoding="utf-8"))["args"]
    parent = ROOT / "checkpoints/sft_v3/epoch_02"
    settings = argparse.Namespace(**{**source_settings, "epochs": options.max_epochs - 2, "lr": 1e-5,
        "out": str(run_dir), "init_from": str(parent), "data": str(ROOT / "data/train.jsonl")})
    assert not any([settings.perm_kl, settings.anchor_w, settings.p_none, settings.p_none_pair,
                    settings.p_none_distract, settings.p_distract, settings.full_ft, settings.shared_prefix,
                    settings.length_sort, settings.row_budget, settings.pass_tokens_max])
    declaration = {"options": vars(options), "training_args": vars(settings),
                   "data_sha256": digest(ROOT / "data/train.jsonl"),
                   "protocol_sha256": digest(ROOT / "CONTINUATION_PROTOCOL.md"),
                   "source_sha256": {"continue_training.py": digest(__file__),
                                     "official_train.py": digest(ROOT.parent / "kev/kev/train.py"),
                                     "evaluate_model.py": digest(ROOT / "evaluate_model.py")}}
    # Launch flags do not change the training definition.
    declaration["options"] = {k: v for k, v in declaration["options"].items() if k not in {"detach", "resume"}}
    fingerprint = canonical_hash(declaration)
    config_path = run_dir / "continuation_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text())["fingerprint"] == fingerprint
        assert options.resume, "Existing run requires --resume; nothing is overwritten"
    else:
        atomic_json(config_path, {**declaration, "fingerprint": fingerprint,
            "environment": {p: importlib.metadata.version(p) for p in
                ["torch", "transformers", "peft", "triton", "flash-linear-attention"]}})
    state_path = out / "controller.json"
    if state_path.exists():
        ledger = json.loads(state_path.read_text())
    else:
        atomic_json(out / "status.json", {"stage": "baseline_validation"})
        baseline_candidates = []
        for epoch in ([2] if options.probe else [1, 2]):
            checkpoint = ROOT / "checkpoints/sft_v3" / f"epoch_{epoch:02d}"
            score = evaluate(checkpoint, out / f"validation_epoch_{epoch:02d}",
                             1 if options.probe else options.repeats, 2 if options.probe else None)
            baseline_candidates.append({**score, "epoch": epoch, "name": f"epoch_{epoch:02d}"})
        best = max(baseline_candidates, key=lambda r: r["score"])
        ledger = {"fingerprint": fingerprint, "candidates": baseline_candidates, "best": best,
                  "meaningful_best_score": best["score"], "bad_epochs": 0, "last_validated_epoch": 2}
        atomic_json(state_path, ledger)
    subprocess.run([str(STOCK), str(ROOT.parent / "kev_service.py"), "stop"], check=True)
    print("Loading continuation training model; parent", parent, flush=True)
    torch.manual_seed(settings.seed)
    sampler = random.Random(settings.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    completed_epoch, phase_step, records_seen, train_seconds = 2, 0, 0, 0.0
    history, grad_norms = [], []
    resume = None
    complete_snapshots = sorted(p for p in checkpoints.glob("epoch_*") if (p / "training_state.pt").exists())
    if options.resume and complete_snapshots:
        latest = complete_snapshots[-1]
        resume = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=False)
        assert resume["fingerprint"] == fingerprint
        parent = latest
    token = load_tokenizer(settings.base, revision=settings.base_revision)
    model = DecisionModel(settings.base, token, "cuda", lora=settings.lora,
        revision=settings.base_revision, head_dim=settings.head_dim, lora_targets=settings.lora_targets,
        option_isolation=False, special_embeddings=False, dtype=torch.float32)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    meta = Meta(base=settings.base, base_revision=settings.base_revision, lora=settings.lora,
                head_dim=settings.head_dim, weights_dtype="fp32")
    provenance = Checkpoint(str(parent)).warm_start(model, meta)
    requests = train.training_requests(settings, token, None, [])
    assert len(requests) == frozen["counts"]["selected_train_records"] == 7224
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
              {"params": head, "lr": settings.lr}]
    optimizer = torch.optim.AdamW(groups, lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[settings.lr]*2,
        total_steps=settings.epochs*per_epoch, pct_start=.1)
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        optimizer_device(optimizer, "cuda")
        scheduler.load_state_dict(resume["scheduler"])
        completed_epoch, phase_step, records_seen, train_seconds = (resume[k] for k in
            ["epoch", "phase_step", "records_seen", "train_seconds"])
        history, grad_norms = resume["loss_history"], resume["grad_norms"]
        requests = [by_id[key] for key in resume["record_order"]]
        sampler.setstate(resume["sampler_rng"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
        print("Restored optimizer, schedule, random states and record order at cumulative epoch", completed_epoch, flush=True)
    atomic_json(run_dir / "training_config.json", {"args": vars(settings), "init_source": provenance,
        "continuation_fingerprint": fingerprint, "optimizer_reset_at_phase_start": True})
    print(f"trainable params={sum(p.numel() for p in trainable)/1e6:.1f}M; {len(requests)} records; {per_epoch} steps/epoch", flush=True)

    def validate_snapshot(epoch, checkpoint):
        nonlocal ledger
        optimizer.zero_grad(set_to_none=True)
        probe_weights = [p.detach().cpu().clone() for p in trainable] if options.probe else None
        probe_moments = copy.deepcopy(optimizer.state_dict()) if options.probe else None
        if probe_moments:
            for state in probe_moments["state"].values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cpu()
        model.to("cpu")
        optimizer_device(optimizer, "cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        atomic_json(out / "status.json", {"stage": "validation", "epoch": epoch,
            "training_gpu_bytes_after_offload": torch.cuda.memory_allocated(), "checkpoint": str(checkpoint)})
        print(f"Epoch {epoch} model and optimizer offloaded for stock-environment validation; GPU bytes {torch.cuda.memory_allocated()}", flush=True)
        try:
            score = evaluate(checkpoint, out / f"validation_epoch_{epoch:02d}",
                             1 if options.probe else options.repeats, 2 if options.probe else None)
        finally:
            subprocess.run([str(STOCK), str(ROOT.parent / "kev_service.py"), "stop"], check=True)
        candidate = {**score, "epoch": epoch, "name": f"epoch_{epoch:02d}"}
        consider(ledger, candidate, options.min_delta)
        ledger["last_validated_epoch"] = epoch
        atomic_json(state_path, ledger)
        print("Validation selection:", json.dumps({"epoch": epoch, "score": score["score"],
              "best_epoch": ledger["best"]["epoch"], "best_score": ledger["best"]["score"],
              "bad_epochs": ledger["bad_epochs"]}), flush=True)
        model.to("cuda")
        optimizer_device(optimizer, "cuda")
        assert [id(p) for p in model.trainable_parameters()] == parameter_ids
        assert [id(p) for g in optimizer.param_groups for p in g["params"]] == [id(p) for g in groups for p in g["params"]]
        if options.probe:
            assert all(torch.equal(p.detach().cpu(), saved) for p, saved in zip(trainable, probe_weights))
            current = optimizer.state_dict()
            for key, state in current["state"].items():
                for field, value in state.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(value.cpu(), probe_moments["state"][key][field])
            print("Probe: all trainable weights and AdamW states exactly preserved across validation", flush=True)
        model.train()
        return stopping(ledger, epoch, options.min_epochs, options.max_epochs, options.patience)

    reason = None
    if resume and ledger["last_validated_epoch"] < completed_epoch:
        reason = validate_snapshot(completed_epoch, parent)
    else:
        reason = stopping(ledger, completed_epoch, options.min_epochs, options.max_epochs, options.patience)
    peak = torch.cuda.max_memory_allocated()
    for epoch in range(completed_epoch + 1, options.max_epochs + 1):
        if reason:
            break
        atomic_json(out / "status.json", {"stage": "training", "epoch": epoch,
            "phase_step": phase_step, "phase_total_steps": settings.epochs*per_epoch,
            "best_epoch": ledger["best"]["epoch"], "best_score": ledger["best"]["score"]})
        model.train()
        sampler.shuffle(requests)
        terms = Counter()
        norms = []
        epoch_started = time.monotonic()
        for chunk, step_records, end in train.microbatch_plan(requests, settings, 1, 0):
            batch = train.encode_batch(model, token, settings, chunk, epoch - 1)
            variants = sum(v.share for v in batch)
            group_records = step_records * variants / len(chunk)
            for part in train.row_passes(batch, settings.row_budget, settings.shared_prefix):
                loss, values = train.batch_loss(model, settings, part, "cuda", {}, None,
                    torch.autocast("cuda", dtype=torch.bfloat16))
                (loss / group_records).backward()
                terms += values
            terms["n"] += variants
            records_seen += round(variants)
            if end:
                norm = float(torch.nn.utils.clip_grad_norm_(trainable, train.MAX_GRAD_NORM, error_if_nonfinite=True))
                norms.append(norm)
                optimizer.step()
                torch.cuda.synchronize()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                phase_step += 1
                if phase_step % 10 == 0 or options.probe:
                    average = terms["ce"] / terms["n"]
                    history.append({"epoch": epoch, "phase_step": phase_step, "loss": average})
                    print(f"epoch {epoch} step {phase_step}/{settings.epochs*per_epoch} loss {average:.3f} lr {scheduler.get_last_lr()[0]:.3g}", flush=True)
                    terms = Counter()
                    atomic_json(out / "training_progress.json", {"epoch": epoch, "phase_step": phase_step,
                               "loss": average, "records_seen": records_seen})
        elapsed = time.monotonic() - epoch_started
        train_seconds += elapsed
        grad_norms.append({"epoch": epoch, "steps": len(norms), "mean": sum(norms)/len(norms),
                           "max": max(norms), "clipped_steps": sum(n > train.MAX_GRAD_NORM for n in norms)})
        peak = max(peak, torch.cuda.max_memory_allocated())
        assert phase_step == (epoch - 2)*per_epoch and records_seen == (epoch - 2)*len(requests)
        target = checkpoints / f"epoch_{epoch:02d}"
        staging = checkpoints / f"epoch_{epoch:02d}.pending"
        staging.mkdir(exist_ok=False)
        model.lm.save_pretrained(staging)
        snapshot_meta = dataclasses.replace(meta,
            head={key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()},
            extra={"args": vars(settings), "init_source": provenance,
                   "data_sha256": digest(settings.data), "continuation_fingerprint": fingerprint,
                   "snapshot": {"epoch": epoch, "optimizer_steps": 904 + phase_step,
                                "phase_optimizer_steps": phase_step}})
        train.finish_checkpoint(staging, snapshot_meta, token)
        state = {"fingerprint": fingerprint, "epoch": epoch, "phase_step": phase_step,
            "records_seen": records_seen, "train_seconds": train_seconds, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "loss_history": history, "grad_norms": grad_norms,
            "record_order": [r["_meta"]["id"] for r in requests], "sampler_rng": sampler.getstate(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        torch.save(state, staging / "training_state.pt")
        atomic_json(staging / "snapshot.json", {"complete": True, "epoch": epoch, "phase_step": phase_step,
                                                "training_state_sha256": digest(staging / "training_state.pt")})
        staging.rename(target)
        del state, batch, loss
        print(f"Saved cumulative epoch {epoch}: {target}; training {elapsed/60:.2f} minutes", flush=True)
        completed_epoch = epoch
        reason = validate_snapshot(epoch, target)
    reason = reason or "maximum_epochs_reached"
    selected = {"checkpoint": ledger["best"]["checkpoint"], "name": ledger["best"]["name"],
        "selected_epoch": ledger["best"]["epoch"], "selected_validation_qoe": ledger["best"]["score"],
        "last_trained_epoch": completed_epoch, "stop_reason": reason, "candidates": ledger["candidates"],
        "uses_test_for_selection": False, "rule": "Maximum mean validation QoE over repeated fixed validation episodes",
        "original_validation_qoe": -33.73654777768037}
    atomic_json(out / "selected.json", selected)
    metrics = {"epochs_completed": completed_epoch - 2, "cumulative_epochs_completed": completed_epoch,
        "optimizer_steps": phase_step, "cumulative_optimizer_steps": 904 + phase_step,
        "records_seen": records_seen, "requested_records": (completed_epoch-2)*len(requests),
        "rejected_records": 0, "wall_seconds": train_seconds, "grad_norm": grad_norms,
        "peak_device_bytes": peak, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "stop_reason": reason, "optimizer_reset_at_phase_start": True, "loss_history": history,
        "batch": settings.batch, "dtype": settings.dtype, "device": "cuda"}
    atomic_json(run_dir / "training_metrics.json", metrics)
    print("Training stopped; best checkpoint:", json.dumps(selected), flush=True)
    del optimizer, scheduler, model, trainable, head, groups
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(out / "status.json", {"stage": "training_complete", "selected_epoch": selected["selected_epoch"],
                                     "last_trained_epoch": completed_epoch, "stop_reason": reason})
    if options.probe:
        atomic_json(out / "completion.json", {"complete": True, "probe": True, "epochs": 2,
                    "optimizer_steps": phase_step, "weights_and_optimizer_preserved": True})
        return
    subprocess.run([str(STOCK), str(ROOT / "finish_continuation.py"), "--run-name", run_name], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="sft_extend_v1")
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=.005)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe", action="store_true")
    options = parser.parse_args()
    assert Path(options.run_name).name == options.run_name and options.run_name not in {".", ".."}
    assert 3 <= options.min_epochs <= options.max_epochs <= 12 and options.patience >= 1
    assert options.repeats >= 1 and options.min_delta > 0
    if options.probe:
        options.min_epochs = options.max_epochs = 4
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(ROOT.parent / ".cache/huggingface"), HF_HUB_OFFLINE="1",
        TRITON_CACHE_DIR=str(ROOT / ".cache/triton"), XDG_CACHE_HOME=str(ROOT / ".cache"),
        PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    if options.detach:
        command = [sys.executable, str(Path(__file__).resolve()), "--run-name", options.run_name,
            "--min-epochs", str(options.min_epochs), "--max-epochs", str(options.max_epochs),
            "--patience", str(options.patience), "--min-delta", str(options.min_delta),
            "--repeats", str(options.repeats)]
        if options.resume:
            command.append("--resume")
        if options.probe:
            command.append("--probe")
        with (logs / f"{options.run_name}.log").open("a" if options.resume else "x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        (logs / f"{options.run_name}.pid").write_text(str(child.pid)+"\n")
        print("Detached continuation PID", child.pid, "log", logs / f"{options.run_name}.log", flush=True)
        return
    out = ROOT / "results" / options.run_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        print("Continuation already complete:", out)
        return
    with (out / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This continuation is already running")
        try:
            run(options)
        except BaseException as error:
            atomic_json(out / "failure.json", {"error": repr(error), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
