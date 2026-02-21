#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from asteroid.models import ConvTasNet


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input WAV (mono preferred)")
    ap.add_argument("--out", dest="outp", required=True, help="Output WAV (primary/loudest stream)")
    ap.add_argument(
        "--model",
        default="mpariente/ConvTasNet_WHAM_sepclean",
        help="HF model id for asteroid ConvTasNet (pretrained)",
    )
    ap.add_argument("--out-sr", type=int, default=16000)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    wav, sr = torchaudio.load(str(inp))
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono

    # ConvTasNet models are often trained around 8k; start with 8k for separation
    target_sr = 8000
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    # Load pretrained ConvTasNet
    model = ConvTasNet.from_pretrained(args.model)
    model.eval()

    with torch.no_grad():
        # model expects shape [batch, time] or [batch, 1, time] depending on version
        x = wav.squeeze(0).unsqueeze(0)  # [1, T]
        est = model(x)  # typically [1, n_src, T]
        est = est.squeeze(0).cpu().numpy()  # [n_src, T]

    if est.ndim != 2 or est.shape[0] < 2:
        raise RuntimeError(f"Unexpected ConvTasNet output shape: {est.shape}")

    s0 = est[0]
    s1 = est[1]

    primary = s0 if rms(s0) >= rms(s1) else s1

    # Resample back to Whisper-friendly SR
    primary_t = torch.from_numpy(primary).unsqueeze(0)  # [1, T]
    if sr != args.out_sr:
        primary_t = torchaudio.functional.resample(primary_t, sr, args.out_sr)
        sr = args.out_sr

    sf.write(str(outp), primary_t.squeeze(0).numpy(), sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
