"""
CPU tests for the Phase 9 warmstart/re-dream helpers. These cover only the
model-free schedule math plus an eager stub of the Euler tail; GPU behavior is
exercised by examples/warm_probe.py.
"""
import sys

sys.path.insert(0, "examples")

import pytest
import torch

from warmstart import partial_denoise, renoise, resume_index


def test_resume_index_accepts_exact_grid_hits():
    sigmas = torch.tensor([1.0, 0.9, 0.75, 0.3, 0.0])
    assert resume_index(sigmas, 0.9) == 1
    assert resume_index(sigmas, 0.75) == 2
    assert resume_index(sigmas, 0.3) == 3


def test_resume_index_snaps_within_tolerance():
    sigmas = [1.0, 0.9, 0.75, 0.3, 0.0]
    assert resume_index(sigmas, 0.2999999) == 3


@pytest.mark.parametrize("sigma", [0.2, 0.31, 0.5, 0.8])
def test_resume_index_rejects_off_grid_sigma(sigma):
    sigmas = [1.0, 0.9, 0.75, 0.3, 0.0]
    with pytest.raises(ValueError, match="not on scheduler grid"):
        resume_index(sigmas, sigma)


@pytest.mark.parametrize("sigma", [1.0, 0.0])
def test_resume_index_rejects_endpoints(sigma):
    sigmas = [1.0, 0.9, 0.75, 0.3, 0.0]
    with pytest.raises(ValueError, match="mid-schedule"):
        resume_index(sigmas, sigma)


def test_renoise_endpoints():
    x0 = torch.randn(2, 3)
    eps = torch.randn(2, 3)
    assert torch.equal(renoise(x0, eps, 0.0), x0)
    assert torch.equal(renoise(x0, eps, 1.0), eps)


def test_renoise_is_linear_forward_map():
    x0 = torch.tensor([1.0, 3.0, -2.0])
    eps = torch.tensor([5.0, -1.0, 2.0])
    got = renoise(x0, eps, 0.25)
    want = torch.tensor([2.0, 2.0, -1.0])
    assert torch.allclose(got, want)


class _KV:
    def __init__(self):
        self.frozen = False
        self.freeze_calls = []

    def set_frozen(self, is_frozen):
        self.frozen = bool(is_frozen)
        self.freeze_calls.append(self.frozen)


class _StubEngine:
    def __init__(self):
        self.scheduler_sigmas = torch.tensor([1.0, 0.9, 0.75, 0.3, 0.0])
        self.kv_cache = _KV()
        self.calls = []

    def model(self, x, sigma, scale=0.25, offset=0.5, kv_cache=None):
        assert kv_cache is self.kv_cache
        sig = sigma.view(sigma.size(0), sigma.size(1), *([1] * (x.ndim - 2)))
        self.calls.append((self.kv_cache.frozen, float(sigma[0, 0])))
        return x * scale + sig * offset


def _manual_tail(x, sigmas, j, scale=0.25, offset=0.5):
    sigma = x.new_empty((x.size(0), x.size(1)))
    for step_sig, step_dsig in zip(sigmas[j:], sigmas[j:].diff()):
        sig = sigma.fill_(step_sig).view(sigma.size(0), sigma.size(1),
                                         *([1] * (x.ndim - 2)))
        v = x * scale + sig * offset
        x = (x.float() + step_dsig.float() * v.float()).type_as(x)
    return x


def test_partial_denoise_from_zero_matches_full_euler_loop():
    eng = _StubEngine()
    x = torch.randn(1, 1, 2, 3, 4)
    got = partial_denoise(eng, x.clone(), {"scale": 0.125, "offset": 0.75}, j=0)
    want = _manual_tail(x.clone(), eng.scheduler_sigmas, j=0, scale=0.125, offset=0.75)
    assert torch.allclose(got, want)
    assert eng.kv_cache.freeze_calls == [True]
    assert [sig for frozen, sig in eng.calls if frozen] == pytest.approx([1.0, 0.9, 0.75, 0.3])


def test_partial_denoise_last_mid_sigma_runs_one_step():
    eng = _StubEngine()
    x = torch.full((1, 1, 1, 2, 3), 2.0)
    got = partial_denoise(eng, x.clone(), {}, j=3)
    sigma = torch.full((1, 1, 1, 1, 1), 0.3)
    v = x * 0.25 + sigma * 0.5
    want = x + (-0.3) * v
    assert torch.allclose(got, want)
    assert [frozen for frozen, _sig in eng.calls] == [True]
    assert [sig for _frozen, sig in eng.calls] == pytest.approx([0.3])
