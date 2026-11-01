"""A missing video encoder cannot silently produce a successful output path."""

import pytest

from moq3dgs.render import writer


def test_missing_video_backend_fails_before_saving_frames(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError("FFMPEG plugin absent")

    monkeypatch.setattr(writer.imageio, "get_writer", unavailable)
    with pytest.raises(RuntimeError, match="imageio\\[ffmpeg\\]"):
        writer.FrameWriter(tmp_path)
