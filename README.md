# MedSpeech Pipeline

Offline speech-to-structured-medical-data pipeline.

## Current Status

✔ Stage 0 – Audio normalization  
✔ Stage 1 – MLX Whisper transcription (Apple Silicon accelerated)  
⬜ Stage 2 – Structured extraction (LM Studio: Qwen)  
⬜ Stage 3 – Clinical review (MedGemma)  
⬜ Stage 4 – Verification  

## Run

```bash
python -m medspeech.cli samples/example.mp3
