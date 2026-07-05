# uv run --dev pytest examples/test_ae_state.py -v
"""
CPU unit tests for the VAE streaming-state snapshot used by engine.get_state /
load_state (src/ae.py). Loads ae.py standalone (importlib) to avoid the package
__init__ -> gemlite -> CUDA import side effect.
"""
import importlib.util
import pathlib
from collections import namedtuple

import torch

_AE = pathlib.Path(__file__).resolve().parents[1] / "src" / "ae.py"
_spec = importlib.util.spec_from_file_location("ae_standalone", _AE)
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)
clone = ae._clone_state

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def test_clone_is_deep_and_independent():
    wq = torch.ones(3)
    mem = torch.full((2,), 5.0)
    enc = torch.full((4,), 7.0)
    state = {
        "decoder_work_queue": [TWorkItem(wq, 0)],
        "decoder_memory": [None, mem, None],
        "n_frames_decoded": 5,
        "_last_encoder_input_frame": enc,
    }
    snap = {k: clone(v) for k, v in state.items()}
    # mutate every original tensor in place
    state["decoder_work_queue"][0].input_tensor.add_(99)
    state["decoder_memory"][1].add_(99)
    state["_last_encoder_input_frame"].add_(99)
    # snapshot retains the pre-mutation values
    assert snap["decoder_work_queue"][0].input_tensor.eq(1).all()
    assert snap["decoder_memory"][1].eq(5).all()
    assert snap["_last_encoder_input_frame"].eq(7).all()
    assert snap["n_frames_decoded"] == 5


def test_clone_preserves_structure():
    snap = clone([TWorkItem(torch.zeros(2), 3), None, 7])
    assert type(snap[0]).__name__ == "TWorkItem"
    assert snap[0].block_index == 3
    assert snap[1] is None
    assert snap[2] == 7


def test_stream_attrs_match_reset_fields():
    """The snapshotted attributes must be EXACTLY the state StreamingTAEHV.reset() sets,
    so a taehv version bump that adds/renames a streaming buffer fails loudly here."""
    import inspect
    import re
    from taehv import StreamingTAEHV
    src = inspect.getsource(StreamingTAEHV.reset)
    # assignment targets: `self.NAME` immediately followed by ',' or '=' (handles tuple
    # unpacking like `self.a, self.b = ...`); excludes reads like `self.taehv.encoder`.
    assigned = set(re.findall(r"self\.(\w+)\s*(?=[,=])", src))
    assert assigned == set(ae.ChunkedStreamingTAEHV._STREAM_ATTRS), (
        f"reset() state {assigned} != _STREAM_ATTRS "
        f"{set(ae.ChunkedStreamingTAEHV._STREAM_ATTRS)}")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
