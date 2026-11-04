"""Serve only Llama ABR checkpoints with their saved delimiter contract."""
import argparse
import runpy
import sys

from runtime import checkpoint_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    checkpoint_runtime(args.run)
    sys.argv = ["kev.serve", "--run", args.run, "--device", args.device,
                "--host", "127.0.0.1", "--port", str(args.port)]
    runpy.run_module("kev.serve", run_name="__main__")


if __name__ == "__main__":
    main()
