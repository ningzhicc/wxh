"""Check paired JSONL against the actual Kev encoder; never drop/truncate rows."""
import argparse
import json
from pathlib import Path
import sys

from prepare import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path, help="Existing local tokenizer directory")
    parser.add_argument("--max-state", type=int, default=512)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kev"))
    from transformers import AutoTokenizer
    from kev.data import load_records, materialize
    from kev.model import encode, training_context
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), local_files_only=True)
    limits = training_context(args.max_state)
    manifest = json.loads((args.pair_dir / "manifest.json").read_text())
    result = {"tokenizer": str(args.tokenizer.resolve()), "tokenizer_sha256": {
        p.name: digest(p) for p in sorted(args.tokenizer.glob("*")) if p.is_file() and
        p.name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")},
        "checker_sha256": digest(__file__),
        "limits": limits, "limited_preview": manifest["limited_preview"], "variants": {}}
    for variant in ("base", "cost"):
        path = args.pair_dir / f"{variant}.jsonl"
        if digest(path) != manifest["files_sha256"][path.name]:
            raise ValueError("Paired data changed")
        count, state_max, row_max = 0, 0, 0
        for request in load_records(path):
            record = materialize(request)
            encoded = encode(tok, record, max_state=limits["max_state"], max_branch=limits["max_branch"], strict=True)
            if len(encoded["ids"]) > limits["max_packed"]:
                raise ValueError(f"{variant} record {count} exceeds packed budget")
            state_max = max(state_max, encoded["state_tokens"])
            row_max = max(row_max, len(encoded["ids"]))
            count += 1
        if count != manifest["records"]:
            raise ValueError("Record count changed")
        result["variants"][variant] = {"records": count, "max_state_tokens": state_max,
                                        "max_packed_tokens": row_max, "dropped": 0, "truncated": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
