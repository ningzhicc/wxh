"""Download on request, or inspect a local model; prepare a reproducible training command."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path

from runtime import install_delimiters, load_llama_tokenizer
from kev.data import load_records, materialize
from kev.model import encode, training_context

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "abr_finetune/data"
PYTHON = ROOT.parent / "kev/.venv/bin/python"
EXPECTED_DATA_SHA = "e0a154e4fa57274f3755f8e3980a1bd3055035699277cccfc29f5b1c41b8e47a"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--revision")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--run-name", default="llama_1b_v1")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {".", "..", ""}:
        raise SystemExit("Run name must be a single directory name")
    if not 1 <= args.epochs <= 10:
        raise SystemExit("Epoch count must be 1..10")
    base = Path(args.base)
    if not base.is_dir():
        from huggingface_hub import snapshot_download
        try:
            base = Path(snapshot_download(args.base, revision=args.revision,
                cache_dir=str(ROOT.parent / ".cache/huggingface/hub"),
                local_files_only=not args.download,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]))
        except Exception as error:
            raise SystemExit(f"Model unavailable ({type(error).__name__}). Obtain official access and log in locally, "
                             "then use --download; alternatively pass --base /path/to/local/model. Never paste a token into chat.") from None
    base = base.resolve()
    if not list(base.glob("*.safetensors")):
        raise SystemExit("No model weights found; a tokenizer/config alone is insufficient")
    tok = load_llama_tokenizer(str(base), local_files_only=True)
    if sha(DATA / "train.jsonl") != EXPECTED_DATA_SHA:
        raise SystemExit("Frozen training data changed; refusing to silently change the comparison")
    context = training_context(512)
    overflow, maxima, count = [], {"state": 0, "packed": 0}, 0
    for count, request in enumerate(load_records(DATA / "train.jsonl"), 1):
        rec = materialize(request)
        try:
            enc = encode(tok, rec, max_state=context["max_state"], max_branch=context["max_branch"], strict=True)
            maxima["state"] = max(maxima["state"], enc["state_tokens"])
            maxima["packed"] = max(maxima["packed"], len(enc["ids"]))
            if len(enc["ids"]) > context["max_packed"]:
                overflow.append(count)
        except ValueError:
            overflow.append(count)
    if overflow or count != 7224:
        raise SystemExit(f"Expected all 7224 records without truncation: count={count}, overflow={overflow[:20]}")
    out = ROOT / "runs" / args.run_name
    checkpoints = ROOT / "checkpoints" / args.run_name
    if out.exists() or checkpoints.exists():
        raise SystemExit("Run/checkpoint destination already exists; choose another run name")
    command = [str(PYTHON), str(ROOT / "train.py"), "--epoch-checkpoints", str(checkpoints),
        "--base", str(base), "--data", str(DATA / "train.jsonl"), "--out", str(out),
        "--epochs", str(args.epochs), "--lr", "2e-5", "--head_lr", "1e-4", "--lora", "16",
        "--head_dim", "256", "--batch", "2", "--accum", "8", "--device", "cuda",
        "--dtype", "bf16", "--weights_dtype", "fp32", "--checkpointing", "1", "--max_state", "512",
        "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0", "--p_none_pair", "0", "--seed", "666"]
    manifest = {"requested_base": args.base, "resolved_base": str(base), "requested_revision": args.revision,
        "config_sha256": sha(base / "config.json"), "train_sha256": EXPECTED_DATA_SHA,
        "base_weights_sha256": {f.name: sha(f) for f in sorted(base.glob("*.safetensors"))},
        "records": count, "encoding": install_delimiters(tok), "max_tokens": maxima,
        "initialization": "original Llama backbone; fresh LoRA and pointer head; no Kev checkpoint",
        "epochs": args.epochs, "command": command,
        "note": "Fresh head uses a separate 1e-4 learning rate; this is a proposed initial setting, not a validated optimum."}
    target = ROOT / "prepared" / f"{args.run_name}.json"
    target.parent.mkdir(exist_ok=True)
    with target.open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"Prepared {target}; training has NOT started.")
    print(shlex.join(command))


if __name__ == "__main__":
    main()
