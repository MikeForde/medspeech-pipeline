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

------------------------------------------------------------------------

# Architecture (Current)

## Stage 0 --- Audio Preparation

-   Converts input audio to 16 kHz mono WAV
-   Applies gentle loudness normalization

------------------------------------------------------------------------

## Stage 0a --- Noise Reduction (DeepFilterNet)

DeepFilterNet2 is used for speech-focused denoising.

DeepFilterNet consistently improved ASR performance without introducing
the musical artifacts common in source separation models.

------------------------------------------------------------------------

## Stage 0b --- Speech Separation (Demucs)

Demucs is optionally used as a competing-source suppressor:

    demucs --two-stems vocals

It is retained for experimentation but generally adds little measurable
benefit beyond Stage 0a.

------------------------------------------------------------------------

# Separation Experiments (Retired)

## SepFormer (SpeechBrain)

-   Introduced distortion in real-world recordings
-   Degraded Whisper transcription accuracy

## Conv-TasNet (Asteroid)

-   Significant distortion
-   Reduced intelligibility and harmed ASR

These models generalized poorly to chaotic field audio and are retained
only as experimental code.

------------------------------------------------------------------------

# Micromamba Audio Environment

Audio separation tools run inside a dedicated micromamba environment:

    medspeech-audio

This isolates PyTorch-based tools from the main `.venv`.

------------------------------------------------------------------------

# Building/Rebuilding the Audio Environment (Clean Demucs Setup)

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

------------------------------------------------------------------------

# Getting the Environment Restarted 

If Demucs appears to disappear after restarting VS Code or opening a new
terminal session, this is usually because `~/bin` is not on PATH.

To quickly restore everything:

### 1. Source the bootstrap script

``` bash
source ./scripts/bootstrap_audio.sh
```

You must source the script (not just run it) so PATH changes persist.

### 2. Verify Demucs is visible

``` bash
which demucs
demucs -h
```

If this works, Stage 0b will function normally.

------------------------------------------------------------------------

# Permanent Fix (Recommended)

Add this line to your `~/.zshrc`:

``` bash
export PATH="$HOME/bin:$PATH"
```

Then reload:

``` bash
source ~/.zshrc
```

------------------------------------------------------------------------

# Running the Pipeline

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

Each run creates a timestamped directory in `runs/` containing:

-   raw.wav
-   transcript_raw.txt
-   clean_0a.wav
-   clean_0a0b.wav
-   transcript_clean.txt
-   run_meta.json

------------------------------------------------------------------------

# Design Principles

-   Fully offline capable
-   Modular switchable stages
-   Hardware portable (M1 → Intel NUC → Android tablet)
-   Evaluation-first architecture
-   Preserve all intermediate artifacts

------------------------------------------------------------------------

# Author

Mike Forde

Private research & development project exploring structured clinical
extraction from chaotic trauma speech environments.
