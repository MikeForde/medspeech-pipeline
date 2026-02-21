# MedSpeech Pipeline

Offline, modular speech-to-structured-medical-data pipeline for military
trauma and emergency environments.

This project establishes the **pre-LLM audio and transcription stages**
of a larger architecture designed to extract structured clinical data
from spoken battlefield narratives.

------------------------------------------------------------------------

## Current Status

✅ Native Python project (Apple Silicon / M1 tested)\
✅ Audio normalization & resampling (FFmpeg)\
✅ Whisper transcription via MLX (Metal-accelerated)\
✅ Stage toggles for pre-processing comparison\
✅ Run artifact tracking & timing metadata

LLM extraction, clinical review, and verification stages will be added
next.

------------------------------------------------------------------------

## Architecture (Current)

### Stage 0 -- Audio Preparation

-   Converts input audio to:
    -   16 kHz
    -   Mono WAV
-   Applies gentle loudness normalization

### Stage 0a -- Noise Reduction (Stub)

-   Currently pass-through
-   Designed to integrate DeepFilterNet / RNNoise

### Stage 0b -- Speech Separation (Stub)

-   Currently pass-through
-   Designed for future SpeechBrain / SepFormer integration

### Stage 1 -- Transcription

-   MLX Whisper (Apple Silicon accelerated)
-   Vocabulary priming for TCCC terminology
-   Produces baseline and post-processing transcripts for comparison

------------------------------------------------------------------------

## Project Structure

    medspeech-pipeline/
      medspeech/
        cli.py
        audio_io.py
        stage0a_denoise.py
        stage0b_separate.py
        whisper_stage.py
      runs/
      samples/

------------------------------------------------------------------------

## Installation (macOS M1)

### 1. Create virtual environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install dependencies

``` bash
pip install typer rich pydantic soundfile numpy requests openai mlx-whisper
brew install ffmpeg
```

------------------------------------------------------------------------

## Running the Pipeline

Place test audio in the `samples/` directory.

Basic run:

``` bash
python -m medspeech.cli samples/test.mp3
```

Bypass pre-processing stages:

``` bash
python -m medspeech.cli samples/test.mp3 --no-0a --no-0b
```

Use explicit Whisper model:

``` bash
python -m medspeech.cli samples/test.mp3 --whisper-model mlx-community/whisper-large-v3-turbo
```

------------------------------------------------------------------------

## Output

Each run creates a timestamped directory inside `runs/` containing:

-   `raw.wav`
-   `transcript_raw.txt`
-   `clean_0a.wav` (if enabled)
-   `clean_0a0b.wav` (if enabled)
-   `transcript_clean.txt`
-   `run_meta.json` (timings + configuration)

This enables empirical comparison of: - Raw vs processed audio - Impact
of denoising/separation on ASR quality

------------------------------------------------------------------------

## Next Steps

-   Integrate DeepFilterNet2 for real Stage 0a denoising
-   Add optional speech separation model (Stage 0b)
-   Connect LM Studio for:
    -   Structured extraction (Qwen 2.5)
    -   Clinical plausibility review (MedGemma)
    -   Verification pass

------------------------------------------------------------------------

## Design Goals

-   Fully offline capable
-   Modular stages (switchable)
-   Hardware-portable (M1 → Intel NUC → Android tablet)
-   Clinically structured output (AF3899L schema planned)
-   Evaluation-first architecture (store all intermediate artifacts)

------------------------------------------------------------------------

## Author

Mike Forde

Private research & development project exploring structured clinical
extraction from chaotic trauma speech environments.
