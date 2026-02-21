import json
import time
from pathlib import Path
import typer
from rich import print

from .audio_io import ensure_dir, normalize_and_resample_to_wav
from .stage0a_denoise import denoise_0a
from .stage0b_separate import separate_0b
from .whisper_stage import transcribe_whisper_mlx

app = typer.Typer(add_completion=False)


def _timestamp_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


@app.command()
def run(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    no_0a: bool = typer.Option(False, "--no-0a", help="Bypass Stage 0a (denoise)."),
    no_0b: bool = typer.Option(False, "--no-0b", help="Bypass Stage 0b (separation)."),
    whisper_model: str = typer.Option("mlx-community/whisper-large-v3-turbo", "--whisper-model"),
    language: str = typer.Option("en", "--lang"),
):
    run_id = _timestamp_id()
    out_dir = runs_dir / run_id
    ensure_dir(out_dir)

    meta = {
        "run_id": run_id,
        "input_audio": str(audio),
        "toggles": {"no_0a": no_0a, "no_0b": no_0b},
        "whisper": {"model": whisper_model, "language": language},
        "timings_sec": {},
        "artifacts": {},
    }

    t0 = time.time()

    # --- Stage: normalize/resample ---
    raw_wav = out_dir / "raw.wav"
    t = time.time()
    normalize_and_resample_to_wav(audio, raw_wav)
    meta["timings_sec"]["prep_normalize_resample"] = round(time.time() - t, 3)
    meta["artifacts"]["raw_wav"] = str(raw_wav)

    # --- Stage 1 baseline transcript (raw) ---
    t = time.time()
    transcript_raw = transcribe_whisper_mlx(raw_wav, model=whisper_model, language=language)
    meta["timings_sec"]["whisper_raw"] = round(time.time() - t, 3)
    (out_dir / "transcript_raw.txt").write_text(transcript_raw + "\n", encoding="utf-8")
    meta["artifacts"]["transcript_raw"] = str(out_dir / "transcript_raw.txt")

    current_wav = raw_wav

    # --- Stage 0a ---
    if not no_0a:
        t = time.time()
        clean_0a = out_dir / "clean_0a.wav"
        denoise_0a(current_wav, clean_0a)
        meta["timings_sec"]["stage_0a"] = round(time.time() - t, 3)
        meta["artifacts"]["clean_0a_wav"] = str(clean_0a)
        current_wav = clean_0a
    else:
        meta["timings_sec"]["stage_0a"] = 0.0

    # --- Stage 0b ---
    if not no_0b:
        t = time.time()
        clean_0a0b = out_dir / "clean_0a0b.wav"
        separate_0b(current_wav, clean_0a0b)
        meta["timings_sec"]["stage_0b"] = round(time.time() - t, 3)
        meta["artifacts"]["clean_0a0b_wav"] = str(clean_0a0b)
        current_wav = clean_0a0b
    else:
        meta["timings_sec"]["stage_0b"] = 0.0

    # --- Stage 1 transcript after preprocessing (if any) ---
    if (not no_0a) or (not no_0b):
        t = time.time()
        transcript_clean = transcribe_whisper_mlx(current_wav, model=whisper_model, language=language)
        meta["timings_sec"]["whisper_clean"] = round(time.time() - t, 3)
        (out_dir / "transcript_clean.txt").write_text(transcript_clean + "\n", encoding="utf-8")
        meta["artifacts"]["transcript_clean"] = str(out_dir / "transcript_clean.txt")

    meta["timings_sec"]["total"] = round(time.time() - t0, 3)

    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[bold green]Done[/bold green] → {out_dir}")
    print("Artifacts:")
    for k, v in meta["artifacts"].items():
        print(f"  • {k}: {v}")


if __name__ == "__main__":
    app()
