from pathlib import Path
import shutil


def denoise_0a(in_wav: Path, out_wav: Path) -> Path:
    """
    Stage 0a: Traditional noise reduction / enhancement.

    v0: pass-through stub (copy input to output).
    Replace with DeepFilterNet2 (or RNNoise) once pipeline is stable.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(in_wav, out_wav)
    return out_wav
