from __future__ import annotations

import shutil
from pathlib import Path
from .audio_io import run_cmd

def separate_0b(in_wav: Path, out_wav: Path) -> Path:
    tool = shutil.which("convtasnet_primary")
    if not tool:
        raise RuntimeError("Stage 0b requires `convtasnet_primary` on PATH.")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([tool, "--in", str(in_wav), "--out", str(out_wav), "--out-sr", "16000"])
    return out_wav

