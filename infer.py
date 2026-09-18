#!/usr/bin/env python3
import os
import argparse

import torch
import soundfile as sf


VARIANTS = {
    "noisy": {
        "module": "masking_on_noisy.gtcrn_iva",
        "checkpoint": "./masking_on_noisy/best_model_0121.tar",
    },
    "iva": {
        "module": "masking_on_iva.gtcrn_iva",
        "checkpoint": "./masking_on_iva/best_model_0110.tar",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference for H-GTCRN (masking on noisy or on IVA)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input noisy wav files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save enhanced wav files",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="noisy",
        choices=sorted(VARIANTS.keys()),
        help="Masking variant: noisy (CRM x mixture) or iva (CRM x IVA speech)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults follow --variant)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_enhanced",
        help="Suffix added to output filename before .wav",
    )
    return parser.parse_args()


def load_model(variant: str, checkpoint_path: str, device: torch.device):
    import importlib

    module = importlib.import_module(VARIANTS[variant]["module"])
    model = module.GTCRN_IVA().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def enhance_one_file(model, wav_path: str, out_path: str, device: torch.device):
    noisy, fs = sf.read(wav_path, dtype="float32")
    if fs != 16000:
        raise ValueError(f"Expected 16000 Hz, but got {fs} for file: {wav_path}")

    if noisy.ndim == 1:
        input_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
    else:
        input_tensor = torch.from_numpy(noisy.T).unsqueeze(0).to(device)

    with torch.inference_mode():
        enh = model(input_tensor).cpu().numpy().squeeze()

    sf.write(out_path, enh, fs)


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = args.checkpoint or VARIANTS[args.variant]["checkpoint"]

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.variant, checkpoint, device)

    wav_files = sorted(
        [
            f for f in os.listdir(args.input_dir)
            if f.lower().endswith(".wav")
        ]
    )

    if not wav_files:
        raise FileNotFoundError(f"No wav files found in input_dir: {args.input_dir}")

    for wav_name in wav_files:
        in_path = os.path.join(args.input_dir, wav_name)
        base_name, ext = os.path.splitext(wav_name)
        out_name = f"{base_name}{args.suffix}{ext}"
        out_path = os.path.join(args.output_dir, out_name)

        try:
            enhance_one_file(model, in_path, out_path, device)
            print(f"[OK] {in_path} -> {out_path}")
        except Exception as e:
            print(f"[FAILED] {in_path}: {e}")


if __name__ == "__main__":
    main()
