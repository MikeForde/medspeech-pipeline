from __future__ import annotations

from pathlib import Path
import inspect
import mlx_whisper

DEFAULT_PRIMING = (
    "TCCC, MARCH, MIST, pneumothorax, tension pneumothorax, tourniquet, junctional, "
    "airway, breathing, circulation, hemorrhage, haemorrhage, SpO2, sats, ETCO2, "
    "GCS, AVPU, pulse, respiratory rate, blood pressure, systolic, diastolic, "
    "temperature, hypothermia, analgesia, ketamine, morphine, TXA, tranexamic acid, "
    "ceftriaxone, antibiotics, oxygen, needle decompression, chest seal"
)


def transcribe_whisper_mlx(
    wav_path: Path,
    *,
    model: str = "large-v3-turbo",
    language: str = "en",
    initial_prompt: str = DEFAULT_PRIMING,
) -> str:
    """
    Returns plain text transcript.

    Handles mlx-whisper API differences across versions by introspecting
    supported parameters at runtime.
    """
    sig = inspect.signature(mlx_whisper.transcribe)
    params = sig.parameters

    kwargs = {}

    # Model selector differs by version:
    # - Many versions use: path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
    # - Some versions may expect model_path=...
    if "path_or_hf_repo" in params:
        kwargs["path_or_hf_repo"] = model
    elif "model_path" in params:
        kwargs["model_path"] = model
    else:
        # As a last resort, try to load a model object if load_model exists
        if hasattr(mlx_whisper, "load_model") and "model" in params:
            m = mlx_whisper.load_model(model)
            kwargs["model"] = m
        # If none of these exist, we just don't pass a model selector and rely on defaults.

    # Prompt kw also varies:
    if "initial_prompt" in params:
        kwargs["initial_prompt"] = initial_prompt
    elif "prompt" in params:
        kwargs["prompt"] = initial_prompt

    if "language" in params:
        kwargs["language"] = language

    result = mlx_whisper.transcribe(str(wav_path), **kwargs)

    # mlx-whisper returns dict-like with "text"
    return (result.get("text") or "").strip()
