from __future__ import annotations

import shutil
from pathlib import Path

from .audio_io import run_cmd, resample_wav


def denoise_0a(in_wav: Path, out_wav: Path) -> Path:
    """
    Stage 0a: Noise reduction using DeepFilterNet2 via the `deep-filter` binary.

    Notes:
    - deep-filter currently supports only 48kHz WAV input. :contentReference[oaicite:2]{index=2}
    - We therefore:
        1) resample in_wav -> temp 48k
        2) run deep-filter -> enhanced 48k
        3) resample enhanced 48k -> out_wav (16k for Whisper stage)
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    deep_filter = shutil.which("deep-filter")
    if not deep_filter:
        raise RuntimeError(
            "Stage 0a requires the `deep-filter` binary (DeepFilterNet2).\n"
            "Install it from the DeepFilterNet GitHub releases and ensure it is on your PATH.\n"
            "Then re-run."
        )

    workdir = out_wav.parent
    tmp_in_48k = workdir / "_tmp_0a_in_48k.wav"
    tmp_out_dir = workdir / "_tmp_0a_out"
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 16k -> 48k
    resample_wav(in_wav, tmp_in_48k, sample_rate=48000, channels=1)

    # 2) run deep-filter (writes output into tmp_out_dir, same filename)
    cmd = [
        deep_filter,
        "-o",
        str(tmp_out_dir),
        str(tmp_in_48k),
    ]
    run_cmd(cmd)

    enhanced_48k = tmp_out_dir / tmp_in_48k.name
    if not enhanced_48k.exists():
        # Some versions may produce different naming; fail loudly so we can adjust.
        raise RuntimeError(
            f"`deep-filter` did not produce expected output: {enhanced_48k}\n"
            f"Found: {[p.name for p in tmp_out_dir.glob('*')]} "
        )

    # 3) 48k -> 16k (Whisper-friendly)
    resample_wav(enhanced_48k, out_wav, sample_rate=16000, channels=1)

    # Cleanup best-effort
    try:
        tmp_in_48k.unlink(missing_ok=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)
    except Exception:
        pass

    return out_wav
