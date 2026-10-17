"""Device-resolution fail-safe tests (no GPU required)."""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

torch = pytest.importorskip("torch")

from camloc.perception.device import resolve_device, to_device


def test_cpu_always_cpu():
    assert resolve_device("cpu") == "cpu"


def test_auto_resolves():
    d = resolve_device("auto")
    assert d in ("cpu",) or d.startswith("cuda")


def test_cuda_falls_back_without_gpu():
    if torch.cuda.is_available():
        pytest.skip("GPU present; fallback path not exercised")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert resolve_device("cuda") == "cpu"
        assert any("CUDA is unavailable" in str(x.message) for x in w)


def test_to_device_returns_actual():
    m = torch.nn.Linear(3, 3)
    moved, dev = to_device(m, "cuda")
    expected = "cpu" if not torch.cuda.is_available() else "cuda"
    assert dev.startswith(expected)
    # tensor params actually live on the reported device
    assert next(moved.parameters()).device.type == dev.split(":")[0]
