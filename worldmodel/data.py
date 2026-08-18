from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    if not files:
        raise ValueError(f"no images found under {root}")
    return files


def _fit_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def _fit_depth(path: Path, size: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        im = Image.fromarray(arr, mode="F")
        im = ImageOps.fit(im, (size, size), method=Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    else:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("F")
            im = ImageOps.fit(im, (size, size), method=Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
        if arr.max() > 1.5:
            arr = arr / max(float(arr.max()), 1.0)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi > lo + 1e-6:
        arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _find_depth(depth_root: Path, image: Path, image_root: Path) -> Path | None:
    rel = image.relative_to(image_root)
    candidates = [
        depth_root / rel.with_suffix(".npy"),
        depth_root / rel.with_suffix(".png"),
        depth_root / f"{image.stem}.npy",
        depth_root / f"{image.stem}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@dataclass
class SceneCache:
    rgb: np.ndarray
    depth: np.ndarray | None
    paths: list[str]

    @property
    def n(self) -> int:
        return int(self.rgb.shape[0])


def cache_name(data_dir: str | Path, depth_dir: str | Path | None, image_size: int) -> str:
    payload = f"{Path(data_dir).resolve()}|{Path(depth_dir).resolve() if depth_dir else '-'}|{image_size}"
    return hashlib.sha1(payload.encode("utf8")).hexdigest()[:12]


def build_or_load_cache(
    data_dir: str | Path,
    *,
    depth_dir: str | Path | None,
    image_size: int,
    cache_dir: str | Path,
    progress=None,
) -> SceneCache:
    data_dir = Path(data_dir)
    depth_root = Path(depth_dir) if depth_dir else None
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_name(data_dir, depth_root, image_size)
    rgb_path = cache_dir / f"rgb_{key}.npy"
    dep_path = cache_dir / f"depth_{key}.npy"
    list_path = cache_dir / f"paths_{key}.txt"

    if rgb_path.exists() and list_path.exists() and (depth_root is None or dep_path.exists()):
        rgb = np.load(rgb_path, mmap_mode="r")
        depth = np.load(dep_path, mmap_mode="r") if depth_root is not None else None
        paths = list_path.read_text(encoding="utf8").splitlines()
        return SceneCache(rgb=rgb, depth=depth, paths=paths)

    images = list_images(data_dir)
    rgb_mm = np.lib.format.open_memmap(rgb_path, mode="w+", dtype=np.uint8, shape=(len(images), image_size, image_size, 3))
    dep_mm = None
    if depth_root is not None:
        dep_mm = np.lib.format.open_memmap(dep_path, mode="w+", dtype=np.float16, shape=(len(images), image_size, image_size))

    kept: list[str] = []
    for i, p in enumerate(images):
        rgb_mm[i] = _fit_rgb(p, image_size)
        if dep_mm is not None:
            dp = _find_depth(depth_root, p, data_dir)
            if dp is None:
                raise FileNotFoundError(f"missing depth for {p}; expected matching .npy/.png under {depth_root}")
            dep_mm[i] = _fit_depth(dp, image_size).astype(np.float16)
        kept.append(str(p))
        if progress and (i % 25 == 0 or i + 1 == len(images)):
            progress(i + 1, len(images), str(p))
    rgb_mm.flush()
    if dep_mm is not None:
        dep_mm.flush()
    list_path.write_text("\n".join(kept), encoding="utf8")
    return SceneCache(
        rgb=np.load(rgb_path, mmap_mode="r"),
        depth=np.load(dep_path, mmap_mode="r") if depth_root is not None else None,
        paths=kept,
    )
