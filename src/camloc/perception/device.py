"""Centralized, fail-safe device resolution for the learned components.

Never hardcode `.cuda()`. Always route through `resolve_device()` /
`to_device()` so the pipeline degrades gracefully: if CUDA is requested but
unavailable (no GPU, driver mismatch, OOM at init, a Pascal/FlashAttention
quirk, etc.) it transparently falls back to CPU instead of crashing. A model
moved with `to_device` that fails on the GPU is retried on CPU.
"""
from __future__ import annotations

import warnings
from typing import Optional


def resolve_device(requested: Optional[str] = None) -> str:
    """Return a usable torch device string ('cuda', 'cuda:N', or 'cpu').

    'auto'/None -> cuda if available else cpu. A requested CUDA device that is
    not actually usable falls back to cpu with a warning.
    """
    try:
        import torch
    except Exception:
        return "cpu"

    req = (requested or "auto").lower()
    if req in ("cpu",):
        return "cpu"
    if req in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    # explicit cuda / cuda:N
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn(
                f"device '{requested}' requested but CUDA is unavailable; "
                f"falling back to CPU."
            )
            return "cpu"
        # validate index if given
        if ":" in req:
            try:
                idx = int(req.split(":", 1)[1])
                if idx >= torch.cuda.device_count():
                    warnings.warn(
                        f"cuda:{idx} out of range "
                        f"({torch.cuda.device_count()} devices); using cuda:0."
                    )
                    return "cuda:0"
            except ValueError:
                return "cuda"
        return req
    warnings.warn(f"unknown device '{requested}'; using CPU.")
    return "cpu"


def to_device(module, device: str):
    """Move a torch module/tensor to `device`, falling back to CPU on failure.

    Returns (moved_object, actual_device_str). Any GPU error (OOM, driver,
    unsupported op during the move) triggers a CPU fallback rather than a
    crash.
    """
    try:
        import torch  # noqa
    except Exception:
        return module, "cpu"

    dev = resolve_device(device)
    if dev == "cpu":
        return module.to("cpu"), "cpu"
    try:
        return module.to(dev), dev
    except Exception as exc:  # pragma: no cover - hardware dependent
        warnings.warn(
            f"moving to '{dev}' failed ({exc!r}); falling back to CPU."
        )
        return module.to("cpu"), "cpu"
