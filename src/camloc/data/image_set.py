"""Group the 7 asynchronous camera streams into causal "image sets".

Each camera has its own exposure stamp. We assemble an image set keyed on
the reference camera's stamp: when the reference camera produces a frame, we
pair it with the most recent frame from every other camera whose stamp is
within `sync_window_ms` and <= the reference stamp (strictly causal).

The set is emitted the moment the reference frame arrives, so no future data
is ever used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

from .mcap_source import ImageMsg, StateSample, McapSource


@dataclass
class ImageSet:
    stamp_ns: int                       # reference camera stamp
    images: Dict[str, ImageMsg] = field(default_factory=dict)

    def __contains__(self, cam: str) -> bool:
        return cam in self.images

    def get(self, cam: str) -> Optional[ImageMsg]:
        return self.images.get(cam)


class ImageSetAssembler:
    """Causal assembler turning a message stream into (image_set, states)."""

    def __init__(self, cfg: dict):
        isc = cfg["image_set"]
        self.ref = isc["reference_cam"]
        self.window_ns = int(isc["sync_window_ms"]) * 1_000_000
        self.cameras = list(cfg["dataset"]["cameras"])
        self._latest: Dict[str, ImageMsg] = {}

    def stream(
        self, source: McapSource, start_ns=None, end_ns=None
    ) -> Iterator[tuple]:
        """Yield ('state', StateSample) and ('set', ImageSet) in time order.

        State samples pass through immediately (the filter consumes them for
        prediction). An image set is emitted when a reference-camera frame
        arrives, bundling the most recent causal frame of each other camera.
        """
        for kind, msg in source.iter_messages(start_ns, end_ns, include_state=True):
            if kind == "state":
                yield ("state", msg)
                continue
            # image
            self._latest[msg.cam] = msg
            if msg.cam == self.ref:
                iset = ImageSet(stamp_ns=msg.stamp_ns)
                for cam in self.cameras:
                    m = self._latest.get(cam)
                    if m is None:
                        continue
                    # strictly causal + within sync window
                    if 0 <= (msg.stamp_ns - m.stamp_ns) <= self.window_ns:
                        iset.images[cam] = m
                    elif cam == self.ref:
                        iset.images[cam] = m
                yield ("set", iset)
