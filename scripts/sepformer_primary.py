#!/usr/bin/env python3
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from speechbrain.pretrained import SepformerSeparation as Separator



def rms(x: torch.Tensor) -> float:
    # x: [T]
    return float(torch.sqrt(torch.mean(x * x) + 1e-12).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input WAV (any SR, mono preferred)")
    ap.add_argument("--out", dest="outp", required=True, help="Output WAV (primary/loudest speaker)")
    ap.add_argument("--model", default="speechbrain/sepformer-wsj02mix")
    ap.add_argument("--savedir", default=str(Path.home() / ".cache" / "medspeech" / "sepformer-wsj02mix"))
    ap.add_argument("--out-sr", type=int, default=16000)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # WSJ0-2Mix SepFormer commonly expects 8kHz in examples
    target_sr = 8000

    wav, sr = torchaudio.load(str(inp))
    # force mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    # Write a temp 8k wav because separate_file() takes a path
    tmp = outp.parent / "_tmp_sepformer_8k.wav"
    sf.write(str(tmp), wav.squeeze(0).numpy(), sr)

    model = Separator.from_hparams(source=args.model, savedir=args.savedir)
    est = model.separate_file(path=str(tmp))  # shape: [T, N] or [1, T, N] depending on version

    # Normalize shape to [T, N]
    if est.dim() == 3:
        est = est.squeeze(0)

    if est.dim() != 2 or est.shape[1] < 2:
        raise RuntimeError(f"Unexpected SepFormer output shape: {tuple(est.shape)}")

    s0 = est[:, 0].detach().cpu()
    s1 = est[:, 1].detach().cpu()

    primary = s0 if rms(s0) >= rms(s1) else s1

    # Resample to Whisper-friendly output SR
    primary = primary.unsqueeze(0)  # [1, T]
    if sr != args.out_sr:
        primary = torchaudio.functional.resample(primary, sr, args.out_sr)
        sr = args.out_sr

    sf.write(str(outp), primary.squeeze(0).numpy(), sr)

    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
