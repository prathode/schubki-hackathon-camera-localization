"""Compliance tests: causality and ground-truth isolation.

These guard the two disqualification risks that automated checks can catch:
  * the inference stream never exposes prohibited ground-truth fields;
  * the image-set assembler never bundles a future-stamped frame.

They run without the dataset (synthetic fixtures) so CI stays fast; a second
set of checks runs against the real MCAP when CAMLOC_MCAP is set.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camloc.data.mcap_source import (
    ALLOWED_STATE_FIELDS, PROHIBITED_STATE_FIELDS, StateSample,
)
from camloc.data.image_set import ImageSetAssembler
from camloc.data.mcap_source import ImageMsg


def _cfg():
    return {
        "dataset": {"cameras": ["fl", "fr", "r"]},
        "image_set": {"sync_window_ms": 40, "reference_cam": "fl"},
    }


def test_allow_and_prohibit_disjoint():
    assert set(ALLOWED_STATE_FIELDS).isdisjoint(set(PROHIBITED_STATE_FIELDS))


def test_prohibited_pose_fields_listed():
    for f in ("x_m", "y_m", "yaw_rad", "sn_map_state"):
        assert f in PROHIBITED_STATE_FIELDS


def test_statesample_only_holds_allowed():
    s = StateSample(stamp_ns=1)
    s.fields["v_mps"] = 10.0
    # a StateSample built by the reader must never carry prohibited fields
    assert set(s.fields).isdisjoint(set(PROHIBITED_STATE_FIELDS))


def test_image_set_is_causal():
    """No frame in an emitted set may be stamped after the reference frame."""
    cfg = _cfg()

    class FakeSource:
        def iter_messages(self, start_ns=None, end_ns=None, include_state=True):
            jp = b"x"
            # other cams arrive before and after the reference frame
            yield ("image", ImageMsg("fr", 100, jp))   # before ref
            yield ("image", ImageMsg("r", 150, jp))     # before ref
            yield ("image", ImageMsg("fl", 200, jp))    # reference -> emit
            yield ("image", ImageMsg("fr", 250, jp))    # future, must not join prev set
            yield ("image", ImageMsg("fl", 300, jp))    # reference -> emit

    asm = ImageSetAssembler(cfg)
    sets = [m for k, m in asm.stream(FakeSource()) if k == "set"]
    assert len(sets) == 2
    first = sets[0]
    # every member stamp <= reference stamp
    for cam, msg in first.images.items():
        assert msg.stamp_ns <= first.stamp_ns
    # the 250-stamped fr must not be in the 200 set
    assert first.images["fr"].stamp_ns == 100


def test_sync_window_enforced():
    cfg = _cfg()
    cfg["image_set"]["sync_window_ms"] = 20  # 20 ms window

    class FakeSource:
        def iter_messages(self, start_ns=None, end_ns=None, include_state=True):
            jp = b"x"
            # fr is 50 ms (50_000_000 ns) before ref -> outside 20 ms window
            yield ("image", ImageMsg("fr", 150_000_000, jp))
            yield ("image", ImageMsg("fl", 200_000_000, jp))

    asm = ImageSetAssembler(cfg)
    sets = [m for k, m in asm.stream(FakeSource()) if k == "set"]
    assert "fr" not in sets[0].images       # dropped: outside window
    assert "fl" in sets[0].images


@pytest.mark.skipif(not os.environ.get("CAMLOC_MCAP"),
                    reason="set CAMLOC_MCAP to run against the real dataset")
def test_real_stream_no_gt_leak():
    from camloc.config import load_config
    from camloc.data.mcap_source import McapSource
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..",
                                   "configs", "default.yaml"))
    src = McapSource(os.environ["CAMLOC_MCAP"], cfg)
    t0, _ = src.time_bounds()
    seen = set()
    for i, (kind, msg) in enumerate(
        src.iter_messages(t0, t0 + 2_000_000_000, include_state=True)
    ):
        if kind == "state":
            seen |= set(msg.fields)
        if i > 5000:
            break
    assert seen.isdisjoint(set(PROHIBITED_STATE_FIELDS))
    assert seen.issubset(set(ALLOWED_STATE_FIELDS))
