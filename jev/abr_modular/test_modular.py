"""Numerical, information-boundary and resume-I/O tests; no real model training."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import DATA, JEV, verify_data
from losses import decision_loss
from pool_data import BITRATES, decode_state, request_from_state
from regret import proxy_targets, terminal_target
from train import epoch_order, parse_args, make_declaration, VARIANTS
from checkpoints import publish, latest_resume, validate_checkpoint, prune_resume

sys.path.insert(0, str(JEV / "kev"))
from kev.checkpoint import Meta
from kev.data import api_request, materialize


def observation():
    return {"buffer_seconds": 3, "previous_bitrate_kbps": 750,
            "next_chunk_sizes_bytes": {str(b): b * 1000 for b in BITRATES}}


def test_proxy_units_chosen_reward_and_regret():
    # 750kB chosen chunk: observed time 2s, no stall, no switch -> QoE .75.
    target = proxy_targets(observation(), 1, 2, .75)
    assert target["estimated_qoe"]["750"] == .75
    assert min(target["regret_qoe"].values()) == 0
    assert target["regret_qoe"]["4300"] > target["regret_qoe"]["750"]
    assert not target["is_exact_counterfactual_replay"]
    assert target["horizon_chunks"] == 1


def test_proxy_rejects_misaligned_logged_reward():
    with pytest.raises(ValueError, match="inconsistent"):
        proxy_targets(observation(), 1, 2, -7)


def test_reward_gap_distinguishes_equal_CE_errors():
    keys = [str(b) for b in BITRATES]
    target = {"valid": True, "regret_qoe": dict(zip(keys, [0, 1, 10, 0, 0, 0]))}
    mild, severe = torch.zeros(6), torch.zeros(6)
    mild[1], severe[2] = 2, 2
    a, _ = decision_loss(mild, 0, target)
    b, _ = decision_loss(severe, 0, target)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    a, _ = decision_loss(mild, 0, target, enabled=True)
    b, _ = decision_loss(severe, 0, target, enabled=True)
    assert a < b


def test_regret_gradients_and_candidate_permutation_alignment():
    keys = [str(b) for b in BITRATES]
    target = {"valid": True, "regret_qoe": dict(zip(keys, [0, 1, 2, 3, 4, 5]))}
    z = torch.tensor([.3, -.5, .7, 1., -.2, .1], requires_grad=True)
    loss, _ = decision_loss(z, 0, target, enabled=True)
    grad = torch.autograd.grad(loss, z)[0]
    perm = [5, 0, 3, 1, 4, 2]
    rotated = z.detach()[perm].requires_grad_(True)
    alternate, _ = decision_loss(rotated, perm.index(0), target, enabled=True, keys=[keys[i] for i in perm])
    alternate_grad = torch.autograd.grad(alternate, rotated)[0]
    torch.testing.assert_close(loss, alternate)
    torch.testing.assert_close(grad[perm], alternate_grad)
    assert torch.isfinite(grad).all()


def test_terminal_only_masks_extra_loss():
    z = torch.zeros(6, requires_grad=True)
    expected = F.cross_entropy(z[None], torch.tensor([2]))
    got, terms = decision_loss(z, 2, terminal_target(), enabled=True)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert terms["regret_valid"] == 0 and terms["regret"] == 0


def test_independent_flags_and_stable_epoch_order():
    names = set()
    for cost, regret in VARIANTS:
        args = parse_args(["--candidate-cost", str(cost), "--regret-loss", str(regret), "--dry-run"])
        names.add(args.variant)
        assert args.batch * args.accum == 16 and args.epochs == 8
    assert names == {"baseline", "cost", "regret", "both"}
    assert epoch_order(30832, 666, 0) == epoch_order(30832, 666, 0)
    assert epoch_order(100, 666, 0) != epoch_order(100, 666, 1)
    assert sorted(epoch_order(100, 666, 0)) == list(range(100))


def test_resume_keeps_declaration_but_module_change_does_not():
    args = parse_args(["--dry-run"])
    original = make_declaration(args)
    args.resume, args.stop_after_steps, args.save_every = True, 20, 50
    assert make_declaration(args)["fingerprint"] == original["fingerprint"]
    args.regret_loss = 1
    assert make_declaration(args)["fingerprint"] != original["fingerprint"]


def test_prepared_full_population_alignment_and_information_boundary():
    manifest = verify_data()
    assert manifest["counts"] == {"records": 30832, "regret_valid": 30176, "regret_masked": 656}
    assert manifest["teacher_logits_used"] is False
    with (DATA / "base.jsonl").open() as a, (DATA / "cost.jsonl").open() as b:
        first, changed = json.loads(next(a)), json.loads(next(b))
    assert first["questions"]["bitrate"]["label"] == "1200"
    assert first["state"]["recent_bitrates_kbps"] == [750, 750]
    assert first["_meta"]["video_size_row"] == 2
    assert first["state"] == changed["state"]
    assert list(first["questions"]["bitrate"]["criteria"]) == list(map(str, BITRATES))
    first["_supervision"]["test_marker"] = "FORBIDDEN_FUTURE_OUTCOME"
    assert "FORBIDDEN_FUTURE_OUTCOME" not in json.dumps(api_request(first))
    assert "_supervision" not in json.dumps(materialize(first))


def test_atomic_checkpoint_rotation_rng_state_and_corruption(tmp_path):
    class FakeBackbone:
        def save_pretrained(self, path):
            (Path(path) / "adapter_config.json").write_text('{"test_only":true}')
            torch.save({"test_only": torch.tensor([3.])}, Path(path) / "adapter_model.bin")
    class FakeTokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer_config.json").write_text('{"test_only":true}')
    model = SimpleNamespace(lm=FakeBackbone(), head=torch.nn.Linear(2, 2))
    meta = Meta(base="unit-test-only", lora=16)
    run = tmp_path / "run"
    rng = torch.get_rng_state()
    expected_next = torch.rand(3)
    state = {"optimizer": {"moment": torch.tensor([7.])}, "torch_rng": rng, "position": {"step": 3}}
    for step in (1, 2, 3):
        publish(run / "resume", f"step_{step:08d}", model, FakeTokenizer(), meta,
                {"fingerprint": "same", "step": step}, state)
    selected = latest_resume(run, "same")
    assert selected.name == "step_00000003"
    (run / "resume/step_00000009").mkdir()
    assert latest_resume(run, "same") == selected
    restored = torch.load(selected / "training_state.pt", weights_only=True)
    torch.set_rng_state(restored["torch_rng"])
    torch.testing.assert_close(torch.rand(3), expected_next, rtol=0, atol=0)
    torch.testing.assert_close(restored["optimizer"]["moment"], torch.tensor([7.]))
    prune_resume(run, "same")
    assert not (run / "resume/step_00000001").exists()
    assert len(list((run / "resume").glob("step_*/complete.json"))) == 2
    (selected / "head.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="modified"):
        validate_checkpoint(selected, "same")


def test_dry_run_never_calls_training(monkeypatch, capsys):
    import train as entry
    monkeypatch.setattr(sys, "argv", ["train.py", "--dry-run"])
    monkeypatch.setattr(entry, "make_declaration", lambda args: {"test_only": True})
    def forbidden(*args):
        raise AssertionError("Training must not run")
    monkeypatch.setattr(entry, "train", forbidden)
    entry.main()
    result = json.loads(capsys.readouterr().out)
    assert result["training_started"] is False and result["model_loaded"] is False
