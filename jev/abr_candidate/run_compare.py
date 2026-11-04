"""Use the frozen common simulator with explicit candidate input transformation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from features import VERSION, transform_request
from prepare import digest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "abr_compare"))
import compare
from policies import JevHTTP


class CandidateHTTP(JevHTTP):
    candidate_mode = "base"

    def _get(self, path, payload=None):
        if path == "/v1/systemone":
            payload = transform_request(payload, self.candidate_mode)
            self.last_candidate_request = payload
        return super()._get(path, payload)

    def metadata(self):
        return {**super().metadata(), "candidate_mode": self.candidate_mode,
                "candidate_feature_version": VERSION, "state_builder": "legacy_six_observations_with_candidate_wrapper",
                "candidate_wrapper_sha256": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))}}

    def choose(self, context):
        action, detail = super().choose(context)
        detail["candidate_request"] = self.last_candidate_request
        return action, detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate-mode", required=True, choices=["base", "cost"])
    options, remaining = parser.parse_known_args()
    if "run" not in remaining or "--policy" not in remaining or remaining[remaining.index("--policy") + 1] != "jev":
        parser.error("Wrapper requires run --policy jev; use original compare.py for other commands")
    CandidateHTTP.candidate_mode = options.candidate_mode
    # Explicit policy dependency injection; common physics/data/scoring files stay frozen.
    compare.JevHTTP = CandidateHTTP
    sys.argv = [sys.argv[0], *remaining]
    compare.main()
