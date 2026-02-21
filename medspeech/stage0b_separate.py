from __future__ import annotations

import shutil
from pathlib import Path

from .audio_io import run_cmd, resample_wav


def separate_0b(in_wav: Path, out_wav: Path) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    demucs = shutil.which("demucs")
    if not demucs:
        raise RuntimeError("Stage 0b requires `demucs` on PATH.")

    workdir = out_wav.parent
    tmp_in_48k = workdir / "_tmp_0b_in_48k.wav"
    tmp_out_dir = workdir / "_tmp_0b_out"
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    # Demucs works best at 44.1/48k
    resample_wav(in_wav, tmp_in_48k, sample_rate=48000, channels=1)

    # Key change:
    # - two-stems=vocals makes Demucs output "vocals.wav" and "no_vocals.wav"
    # - we then pick vocals.wav deterministically
    cmd = [
        demucs,
        "--two-stems", "vocals",
        "-o", str(tmp_out_dir),
        str(tmp_in_48k),
    ]
    run_cmd(cmd)

    # Deterministic selection: find vocals.wav anywhere under tmp_out_dir
    vocals = list(tmp_out_dir.rglob("vocals.wav"))
    if not vocals:
        raise RuntimeError(
            f"demucs ran but no vocals.wav found under {tmp_out_dir}. "
            f"Found: {[p.name for p in tmp_out_dir.rglob('*.wav')]}"
        )

    # If multiple, choose the newest/largest (usually only one)
    candidate = max(vocals, key=lambda p: p.stat().st_size)

    # Back to Whisper-friendly 16k
    resample_wav(candidate, out_wav, sample_rate=16000, channels=1)

    # ... your cleanup ...
    return out_wav
