from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

from worldmodel.vkitti2 import build_or_load_vkitti2_cache, discover_vkitti2


def _write_pair(root: Path, *, variation: str, frame: int, depth_cm: int) -> None:
    rgb = root / "vkitti_2.0.3_rgb" / "Scene01" / variation / "frames" / "rgb" / "Camera_0" / f"rgb_{frame:05d}.jpg"
    dep = root / "vkitti_2.0.3_depth" / "Scene01" / variation / "frames" / "depth" / "Camera_0" / f"depth_{frame:05d}.png"
    rgb.parent.mkdir(parents=True, exist_ok=True)
    dep.parent.mkdir(parents=True, exist_ok=True)

    arr = np.zeros((12, 24, 3), dtype=np.uint8)
    arr[..., 0] = 80
    arr[..., 1] = 140
    arr[..., 2] = 200
    Image.fromarray(arr).save(rgb)

    depth = np.full((12, 24), depth_cm, dtype=np.uint16)
    Image.fromarray(depth, mode="I;16").save(dep)


def test_default_discovery_pairs_separate_archives_and_excludes_rotated(tmp_path: Path):
    _write_pair(tmp_path, variation="clone", frame=0, depth_cm=1000)
    _write_pair(tmp_path, variation="15-deg-left", frame=0, depth_cm=1000)

    samples = discover_vkitti2(tmp_path)
    assert len(samples) == 1
    assert samples[0].scene == "Scene01"
    assert samples[0].variation == "clone"
    assert samples[0].frame == 0


def test_metric_depth_uses_one_global_scale(tmp_path: Path):
    _write_pair(tmp_path, variation="clone", frame=0, depth_cm=1000)  # 10 m
    _write_pair(tmp_path, variation="clone", frame=1, depth_cm=4000)  # 40 m

    cache = build_or_load_vkitti2_cache(
        tmp_path,
        image_size=8,
        cache_dir=tmp_path / "cache",
        max_depth_m=80.0,
    )
    assert cache.n == 2
    assert float(np.asarray(cache.depth[0]).mean()) == pytest.approx(10.0 / 80.0, abs=2e-3)
    assert float(np.asarray(cache.depth[1]).mean()) == pytest.approx(40.0 / 80.0, abs=2e-3)
