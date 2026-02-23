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
✅ DeepFilterNet integrated (Stage 0a)\
✅ Demucs integrated (Stage 0b -- optional)\
✅ Whisper transcription via MLX (Metal-accelerated)\
✅ Stage toggles for empirical comparison\
✅ Run artifact tracking & timing metadata

⚠️ SepFormer and Conv-TasNet were evaluated but are not currently used
(see "Separation Experiments" below).

------------------------------------------------------------------------

# First-Time Setup (macOS Mx)

## 1. System Dependencies

``` bash
brew install ffmpeg micromamba
```

------------------------------------------------------------------------

## 2. Python Virtual Environment (.venv)

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install typer rich pydantic soundfile numpy requests mlx-whisper deepfilternet
```

DeepFilterNet runs inside the main `.venv`.

------------------------------------------------------------------------

## 3. Micromamba Audio Environment (Demucs)

Demucs runs in a separate micromamba environment to isolate PyTorch
dependencies from the main `.venv`.

Create the environment:

``` bash
micromamba create -n medspeech-audio -c conda-forge python=3.11 -y
micromamba run -n medspeech-audio python -m pip install --upgrade pip
micromamba run -n medspeech-audio python -m pip install demucs torchcodec
```

Create a wrapper so Demucs is callable from the main shell.

Create `~/bin/demucs`:

``` bash
#!/usr/bin/env bash
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
exec /opt/homebrew/bin/micromamba run -n medspeech-audio demucs "$@"
```

Then:

``` bash
chmod +x ~/bin/demucs
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

------------------------------------------------------------------------

# Returning and Re-Running Experiments

After reopening VSCode or starting a new terminal session, run:

``` bash
source scripts/dev_shell.sh
```

This will:

-   Add `~/bin` to PATH
-   Ensure Demucs wrapper exists
-   Activate `.venv`
-   Verify micromamba environment
-   Confirm Demucs is runnable

You are then ready to execute experiments.

------------------------------------------------------------------------

# Architecture (Current)

## Stage 0 --- Audio Preparation

-   Converts input audio to 16 kHz mono WAV
-   Applies gentle loudness normalization

------------------------------------------------------------------------

## Stage 0a --- Noise Reduction (DeepFilterNet)

DeepFilterNet2 is used for speech-focused denoising.

DeepFilterNet consistently improves ASR performance without introducing
the musical artifacts common in source separation models.

------------------------------------------------------------------------

## Stage 0b --- Speech Separation (Demucs)

Demucs is optionally used as a competing-source suppressor:

    demucs --two-stems vocals

Demucs is retained primarily for experimentation in scenarios involving
strong competing structured audio (radio, PA systems, background
speech).

------------------------------------------------------------------------

# Separation Experiments (Retired)

## SepFormer (SpeechBrain)

-   Model: speechbrain/sepformer-wsj02mix
-   Transformer-based two-speaker separation
-   Introduced distortion in real-world recordings
-   Degraded Whisper transcription accuracy

## Conv-TasNet (Asteroid)

-   Model: mpariente/ConvTasNet_WHAM_sepclean
-   Convolutional time-domain separation network
-   Produced significant distortion
-   Reduced intelligibility and harmed ASR

These models generalized poorly to chaotic field audio and are retained
only as experimental code.

------------------------------------------------------------------------

# Running the Pipeline

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

# Output

Each run creates a timestamped directory inside `runs/` containing:

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

1.  Strong denoising\
2.  Optimized Whisper decoding\
3.  Downstream structured extraction + verification

------------------------------------------------------------------------

# Author

Mike Forde

Private research & development project exploring structured clinical
extraction from chaotic trauma speech environments.
