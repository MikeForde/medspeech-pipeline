# MedSpeech Pipeline

Offline, modular speech-to-structured-medical-data pipeline for military
trauma and emergency environments.

This project establishes the **pre-LLM audio and transcription stages**
of a larger architecture designed to extract structured clinical data
from spoken battlefield narratives.

------------------------------------------------------------------------

# Current Status

✅ Native Python project (Apple Silicon / M1 tested)\
✅ Audio normalization & resampling (FFmpeg)\
✅ Stage toggles for pre-processing comparison\
✅ DeepFilterNet integrated (Stage 0a)\
✅ Demucs integrated (Stage 0b -- optional)\
✅ Whisper transcription via MLX (Metal-accelerated)\
✅ Run artifact tracking & timing metadata

⚠️ SepFormer and Conv-TasNet were evaluated but are not currently used
(see below).

LLM extraction, clinical review, and verification stages will be added
next.

------------------------------------------------------------------------

# Architecture (Current)

## Stage 0 --- Audio Preparation

-   Converts input audio to:
    -   16 kHz
    -   Mono WAV
-   Applies gentle loudness normalization

------------------------------------------------------------------------

## Stage 0a --- Noise Reduction (DeepFilterNet)

DeepFilterNet2 is used for speech-focused denoising.

### Why DeepFilterNet?

-   Designed specifically for speech enhancement
-   Performs well on:
    -   Background environmental noise
    -   Handling noise
    -   Broadband interference
-   Minimal speech distortion
-   Fast enough for laptop deployment

In testing, DeepFilterNet consistently improved ASR performance without
introducing the musical artifacts common in source separation models.

------------------------------------------------------------------------

## Stage 0b --- Speech Separation (Demucs)

Demucs is optionally used as a **competing-source suppressor** using:

    demucs --two-stems vocals

### Observations

-   Demucs performs reasonably well at isolating vocal-like content.
-   It does **not significantly improve ASR results** when DeepFilterNet
    has already been applied.
-   It takes longer to run than DeepFilterNet.
-   In most test cases, it did **not add measurable benefit** beyond
    Stage 0a.

Demucs is therefore kept optional and primarily for experimentation in
scenarios involving strong competing structured audio (e.g., radio,
background speech, PA systems).

------------------------------------------------------------------------

# Separation Experiments (Retired)

Two advanced speech separation models were evaluated:

## 1️⃣ SepFormer (SpeechBrain)

-   Model: speechbrain/sepformer-wsj02mix
-   Single-channel two-speaker separation
-   Transformer-based architecture

### Result

-   Introduced distortion in real-world recordings
-   Degraded Whisper transcription accuracy
-   Produced artifacts worse than background noise

Conclusion: Not suitable for chaotic field audio in current form.

------------------------------------------------------------------------

## 2️⃣ Conv-TasNet (Asteroid)

-   Model: mpariente/ConvTasNet_WHAM_sepclean
-   Convolutional time-domain separation network

### Result

-   Significant distortion
-   Reduced intelligibility
-   Worse ASR performance than DeepFilterNet alone

Conclusion: Also not suitable for this operational domain.

------------------------------------------------------------------------

## Why Separation Failed

Most pretrained separation models are trained on: - Clean synthetic
mixtures (WSJ0-2Mix, WHAM) - Telephone-bandwidth speech - Limited
environmental variability

Battlefield-style recordings contain: - Reverb - Mic coloration -
Non-stationary environmental noise - Irregular overlap

These models generalized poorly to this domain and introduced artifacts
that harmed downstream ASR.

------------------------------------------------------------------------

# Micromamba Audio Environment

Separation models required PyTorch-based environments separate from the
main `.venv`.

We use a dedicated micromamba environment:

    medspeech-audio

## Rebuilding the Environment (Clean Demucs Setup)

Over time, multiple experiments (SepFormer, Conv-TasNet) introduced
dependency conflicts. The clean setup is:

### 1. Remove old environment

``` bash
micromamba remove -n medspeech-audio --all -y
```

### 2. Recreate environment

``` bash
micromamba create -n medspeech-audio -c conda-forge python=3.11 -y
```

### 3. Install Demucs

``` bash
micromamba run -n medspeech-audio python -m pip install --upgrade pip
micromamba run -n medspeech-audio python -m pip install demucs torchcodec
```

### 4. Create wrapper

Create \~/bin/demucs:

``` bash
#!/usr/bin/env bash
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
exec micromamba run -n medspeech-audio demucs "$@"
```

Then:

``` bash
chmod +x ~/bin/demucs
export PATH="$HOME/bin:$PATH"
```

------------------------------------------------------------------------

# Project Structure

    medspeech-pipeline/
      medspeech/
        cli.py
        audio_io.py
        stage0a_denoise.py
        stage0b_separate.py
        whisper_stage.py
      scripts/        # experimental separation models (retained but unused)
      runs/
      samples/

------------------------------------------------------------------------

# Installation (macOS M1)

## 1. Create virtual environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Install dependencies

``` bash
pip install typer rich pydantic soundfile numpy requests mlx-whisper
brew install ffmpeg
```

------------------------------------------------------------------------

# Running the Pipeline

Place test audio in the samples/ directory.

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

# Output

Each run creates a timestamped directory inside runs/ containing:

-   raw.wav
-   transcript_raw.txt
-   clean_0a.wav
-   clean_0a0b.wav
-   transcript_clean.txt
-   run_meta.json

This enables empirical comparison of:

-   Raw vs processed audio
-   Impact of denoising/separation on ASR quality
-   Timing of each stage

------------------------------------------------------------------------

# Design Principles

-   Fully offline capable
-   Modular switchable stages
-   Hardware portable (M1 → Intel NUC → Android tablet)
-   Evaluation-first architecture
-   Preserve all intermediate artifacts
-   Empirical over theoretical model choice

------------------------------------------------------------------------

# Current Conclusion

DeepFilterNet (Stage 0a) provides the most reliable improvement for
real-world trauma speech audio.

Advanced source separation (SepFormer, Conv-TasNet) introduced artifacts
that degraded ASR performance and are therefore retained only as
experimental code.

Demucs is optional but does not materially improve performance when
DeepFilterNet is already applied.

The pipeline will therefore prioritize:

1.  Strong denoising
2.  Optimized Whisper decoding
3.  Downstream structured extraction + verification

------------------------------------------------------------------------

# Author

Mike Forde

Private research & development project exploring structured clinical
extraction from chaotic trauma speech environments.
