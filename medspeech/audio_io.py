import subprocess
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}\n"
        )

# More verbose version of above that streams output live, but doesn't capture it for error reporting.
# def run_cmd(cmd: list[str]) -> None:
#     # stream output live so long-running tools don't look "hung"
#     proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#     assert proc.stdout is not None
#     for line in proc.stdout:
#         print(line, end="")
#     rc = proc.wait()
#     if rc != 0:
#         raise RuntimeError(f"Command failed (exit {rc}): {' '.join(cmd)}")


def normalize_and_resample_to_wav(
    in_audio: Path,
    out_wav: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Convert any input audio to a Whisper-friendly mono WAV at 16kHz.
    Applies gentle loudness normalization to reduce amplitude variance.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # -af: loudnorm gives consistent levels across recordings.
    # -ar/-ac: force sample rate + mono.
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_audio),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-af",
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        str(out_wav),
    ]
    run_cmd(cmd)
    return out_wav

def resample_wav(
    in_wav: Path,
    out_wav: Path,
    *,
    sample_rate: int,
    channels: int = 1,
) -> Path:
    """Resample WAV without changing loudness."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(out_wav),
    ]
    run_cmd(cmd)
    return out_wav

