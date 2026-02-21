from __future__ import annotations

import shutil
from pathlib import Path

from .audio_io import run_cmd


def separate_0b(in_wav: Path, out_wav: Path) -> Path:
    """
    Stage 0b: Overlapping-speaker separation (2-speaker) using SepFormer.
    Keeps the louder stream as "primary voice".
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    tool = shutil.which("sepformer_primary")
    if not tool:
        raise RuntimeError(
            "Stage 0b requires `sepformer_primary` on PATH.\n"
            "Create it in ~/bin as per setup instructions."
        )

    cmd = [
        tool,
        "--in", str(in_wav),
        "--out", str(out_wav),
        "--out-sr", "16000",
        # model default is sepformer-wsj02mix; override here if you want
    ]
    run_cmd(cmd)
    return out_wav
