from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image, ImageOps

from .data import SceneCache


VKITTI2_SCENES = ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20")
VKITTI2_DEFAULT_VARIATIONS = ("clone", "morning", "overcast", "rain", "fog", "sunset")
VKITTI2_ROTATED_VARIATIONS = ("15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right")


@dataclass(frozen=True)
class VKitti2Sample:
    scene: str
    variation: str
    frame: int
    camera: int
    rgb_path: Path
    depth_path: Path


def _describe_path(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for i, part in enumerate(parts):
        if part in VKITTI2_SCENES and i + 1 < len(parts):
            return part, parts[i + 1]
    return None


def _frame_index(path: Path, prefix: str) -> int | None:
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    try:
        return int(stem[len(prefix):])
    except ValueError:
        return None


def discover_vkitti2(
    root: str | Path,
    *,
    camera: int = 0,
    variations: tuple[str, ...] = VKITTI2_DEFAULT_VARIATIONS,
) -> list[VKitti2Sample]:
    """Find paired RGB/depth frames under an extracted Virtual KITTI 2 root.

    `root` should normally be the common parent containing the extracted RGB and
    depth archives. The two archives may be separate directory trees; pairing is
    by (scene, variation, frame, camera), not by relative path.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    cam_name = f"Camera_{camera}"
    wanted = set(variations)
    rgb: dict[tuple[str, str, int], Path] = {}
    depth: dict[tuple[str, str, int], Path] = {}

    for p in root.rglob("rgb_*.jpg"):
        if p.parent.name != cam_name or p.parent.parent.name.lower() != "rgb":
            continue
        sv = _describe_path(p)
        frame = _frame_index(p, "rgb_")
        if sv is None or frame is None or sv[1] not in wanted:
            continue
        rgb[(sv[0], sv[1], frame)] = p

    for p in root.rglob("depth_*.png"):
        if p.parent.name != cam_name or p.parent.parent.name.lower() != "depth":
            continue
        sv = _describe_path(p)
        frame = _frame_index(p, "depth_")
        if sv is None or frame is None or sv[1] not in wanted:
            continue
        depth[(sv[0], sv[1], frame)] = p

    if not rgb:
        raise ValueError(
            "no Virtual KITTI 2 RGB frames found. Choose the common parent containing the extracted RGB archive."
        )
    if not depth:
        raise ValueError(
            "no Virtual KITTI 2 depth frames found. Extract the depth archive too, then choose the common parent containing both RGB and depth."
        )

    keys = sorted(set(rgb) & set(depth))
    if not keys:
        raise ValueError("Virtual KITTI 2 RGB and depth were found, but no frames could be paired.")

    missing_depth = len(set(rgb) - set(depth))
    if missing_depth:
        raise ValueError(
            f"{missing_depth} selected Virtual KITTI 2 RGB frames have no matching depth frame. "
            "Make sure RGB and depth archives are the same dataset version and are both under the selected root."
        )

    return [
        VKitti2Sample(
            scene=scene,
            variation=variation,
            frame=frame,
            camera=camera,
            rgb_path=rgb[(scene, variation, frame)],
            depth_path=depth[(scene, variation, frame)],
        )
        for scene, variation, frame in keys
    ]


def _fit_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def _fit_vkitti_depth(path: Path, size: int, *, max_depth_m: float) -> np.ndarray:
    if max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive")
    with Image.open(path) as im:
        arr = np.asarray(im, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected single-channel VKITTI2 depth PNG, got shape {arr.shape} from {path}")

    # VKITTI2 stores depth in centimetres: integer value 1 == 1 cm.
    depth_m = arr * 0.01
    unit = np.clip(depth_m / float(max_depth_m), 0.0, 1.0).astype(np.float32)
    # Use the same center-square crop as RGB. This is intentional for v0: the
    # renderer is square. v1 should become pose-aware and support native aspect.
    im = Image.fromarray(unit, mode="F")
    im = ImageOps.fit(im, (size, size), method=Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32).clip(0.0, 1.0)


def _cache_key(root: Path, image_size: int, max_depth_m: float, camera: int, variations: tuple[str, ...]) -> str:
    payload = f"vkitti2|{root.resolve()}|{image_size}|{max_depth_m:.6g}|{camera}|{','.join(variations)}"
    return hashlib.sha1(payload.encode("utf8")).hexdigest()[:12]


def build_or_load_vkitti2_cache(
    root: str | Path,
    *,
    image_size: int,
    cache_dir: str | Path,
    max_depth_m: float = 80.0,
    camera: int = 0,
    variations: tuple[str, ...] = VKITTI2_DEFAULT_VARIATIONS,
    progress=None,
) -> SceneCache:
    """Build a fixed-scale RGB + metric-depth cache for Virtual KITTI 2.

    Unlike the generic depth loader, depth is NOT percentile-normalised per
    image. A depth of 10 m must mean the same thing in every scene. Values are
    converted from centimetres and divided by `max_depth_m` once globally.
    """
    root = Path(root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(root, image_size, max_depth_m, camera, variations)
    rgb_path = cache_dir / f"vkitti2_rgb_{key}.npy"
    dep_path = cache_dir / f"vkitti2_depth_{key}.npy"
    list_path = cache_dir / f"vkitti2_paths_{key}.txt"

    if rgb_path.exists() and dep_path.exists() and list_path.exists():
        return SceneCache(
            rgb=np.load(rgb_path, mmap_mode="r"),
            depth=np.load(dep_path, mmap_mode="r"),
            paths=list_path.read_text(encoding="utf8").splitlines(),
        )

    samples = discover_vkitti2(root, camera=camera, variations=variations)
    rgb_mm = np.lib.format.open_memmap(
        rgb_path, mode="w+", dtype=np.uint8,
        shape=(len(samples), image_size, image_size, 3),
    )
    dep_mm = np.lib.format.open_memmap(
        dep_path, mode="w+", dtype=np.float16,
        shape=(len(samples), image_size, image_size),
    )

    paths: list[str] = []
    for i, sample in enumerate(samples):
        rgb_mm[i] = _fit_rgb(sample.rgb_path, image_size)
        dep_mm[i] = _fit_vkitti_depth(sample.depth_path, image_size, max_depth_m=max_depth_m).astype(np.float16)
        paths.append(
            f"{sample.scene}/{sample.variation}/Camera_{sample.camera}/{sample.frame:05d}|{sample.rgb_path}|{sample.depth_path}"
        )
        if progress and (i % 25 == 0 or i + 1 == len(samples)):
            progress(i + 1, len(samples), str(sample.rgb_path))

    rgb_mm.flush()
    dep_mm.flush()
    list_path.write_text("\n".join(paths), encoding="utf8")
    return SceneCache(
        rgb=np.load(rgb_path, mmap_mode="r"),
        depth=np.load(dep_path, mmap_mode="r"),
        paths=paths,
    )
