from fractions import Fraction
from itertools import pairwise

import av
import numpy as np

from swisher.session import inspect_video, normalize_browser_video


def test_webm_recording_becomes_indexed_h264_with_timestamps(tmp_path):
    source, target = tmp_path / "recording.webm", tmp_path / "indexed.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libvpx", rate=30)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for index in range(15):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), index * 10, dtype=np.uint8), format="rgb24"
            )
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    original = source.read_bytes()
    normalize_browser_video(source, target)
    assert source.read_bytes() == original
    assert inspect_video(target)["frame_count"] == 15
    with av.open(str(target)) as container:
        times = [frame.time for frame in container.decode(video=0)]
    assert all(a < b for a, b in pairwise(times))
    assert abs(times[-1] - 14 / 30) < 0.01
