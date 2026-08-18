from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalise_inverse_depth(raw: np.ndarray, *, invert: bool = True) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    raw = np.nan_to_num(raw)
    lo, hi = np.percentile(raw, [2.0, 98.0])
    x = np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    # Relative Depth Anything V2 output is disparity-like in the official
    # release. WorldSplat stores 0=near, 1=far.
    if invert:
        x = 1.0 - x
    return x.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute relative depth maps for WorldSplat using a Transformers depth-estimation model.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--no-invert", action="store_true", help="do not convert disparity-like output to 0-near/1-far")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu; default lets Transformers choose")
    args = ap.parse_args()

    try:
        from transformers import pipeline
    except Exception as e:
        raise SystemExit("Install transformers first: pip install transformers accelerate") from e

    root = Path(args.data)
    out = Path(args.out)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"no images under {root}")
    out.mkdir(parents=True, exist_ok=True)

    kwargs = {"task": "depth-estimation", "model": args.model}
    if args.device is not None:
        kwargs["device"] = args.device
    pipe = pipeline(**kwargs)

    for i, p in enumerate(files, 1):
        rel = p.relative_to(root)
        dst = (out / rel).with_suffix(".npy")
        png = (out / rel).with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            result = pipe(im)
        raw = result.get("predicted_depth")
        if raw is None:
            raw = np.asarray(result["depth"], dtype=np.float32)
        elif hasattr(raw, "detach"):
            raw = raw.detach().float().cpu().numpy()
        raw = np.squeeze(np.asarray(raw, dtype=np.float32))
        d = normalise_inverse_depth(raw, invert=not args.no_invert)
        np.save(dst, d.astype(np.float16))
        Image.fromarray((d * 255).astype(np.uint8)).save(png)
        print(f"{i}/{len(files)} {p} -> {dst}")


if __name__ == "__main__":
    main()
