import copy
import torch

from continue_training import consider, stopping, optimizer_device


def test_small_improvements_preserve_best_but_do_not_extend_patience_forever():
    ledger = {"candidates": [], "best": {"score": 1.0, "epoch": 2},
              "meaningful_best_score": 1.0, "bad_epochs": 0}
    for epoch, score in [(3, 1.002), (4, .98), (5, 1.004)]:
        consider(ledger, {"score": score, "epoch": epoch}, .005)
    assert ledger["best"]["epoch"] == 5 and ledger["bad_epochs"] == 3
    assert stopping(ledger, 5, 6, 10, 3) is None
    consider(ledger, {"score": 1.003, "epoch": 6}, .005)
    assert stopping(ledger, 6, 6, 10, 3) == "validation_plateau"
    assert ledger["best"]["epoch"] == 5


def test_substantial_improvement_resets_patience_and_ties_keep_earlier_checkpoint():
    ledger = {"candidates": [], "best": {"score": 1.0, "epoch": 2},
              "meaningful_best_score": 1.0, "bad_epochs": 2}
    consider(ledger, {"score": 1.02, "epoch": 3}, .005)
    assert ledger["bad_epochs"] == 0 and ledger["meaningful_best_score"] == 1.02
    consider(ledger, {"score": 1.02, "epoch": 4}, .005)
    assert ledger["best"]["epoch"] == 3
    assert stopping(ledger, 10, 6, 10, 3) == "maximum_epochs_reached"


def test_optimizer_transfer_preserves_moments_and_cpu_step_counters():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    parameter.sum().backward()
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    optimizer_device(optimizer, "cpu")
    current = optimizer.state_dict()
    for key, state in current["state"].items():
        for field, value in state.items():
            assert torch.equal(value, saved["state"][key][field])
    assert optimizer.state[parameter]["step"].device.type == "cpu"
