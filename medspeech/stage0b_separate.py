from pathlib import Path
import shutil


def separate_0b(in_wav: Path, out_wav: Path) -> Path:
    """
    Stage 0b: Speech–noise separation.

    v0: pass-through stub.
    Later: replace with a real separation/enhancement model.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(in_wav, out_wav)
    return out_wav
