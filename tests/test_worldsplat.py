from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image

from worldmodel.worldsplat import WorldSplatConfig, WorldSplatVAE, SplatDecoder, _anchor_grid
from worldmodel.train import TrainConfig, train_worldsplat


def test_renderer_shapes_and_gradients():
    cfg = WorldSplatConfig(image_size=24, num_splats=9, latent_dim=4, hidden_dim=32)
    model = WorldSplatVAE(cfg)
    z = torch.randn(2, 4, requires_grad=True)
    out = model.render_latent(z, yaw_deg=8.0)
    assert out.rgb.shape == (2, 3, 24, 24)
    assert out.depth.shape == (2, 24, 24)
    assert out.coverage.shape == (2, 24, 24)
    assert torch.isfinite(out.rgb).all()
    out.rgb.mean().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_anchor_grid_512_is_symmetric_and_full_extent():
    pts = _anchor_grid(512)
    assert pts.shape == (512, 2)
    assert torch.unique(pts, dim=0).shape[0] == 512
    assert torch.allclose(pts.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.isclose(pts[:, 0].min(), torch.tensor(-0.85), atol=1e-6)
    assert torch.isclose(pts[:, 0].max(), torch.tensor(+0.85), atol=1e-6)
    assert torch.isclose(pts[:, 1].min(), torch.tensor(-0.85), atol=1e-6)
    assert torch.isclose(pts[:, 1].max(), torch.tensor(+0.85), atol=1e-6)


def test_ray_screen_position_is_independent_of_depth():
    cfg = WorldSplatConfig(image_size=24, num_splats=9, latent_dim=4, hidden_dim=32, focal=1.25)
    dec = SplatDecoder(cfg)

    # Same learned ray offsets, radically different depth logits.
    raw = torch.zeros(2, cfg.num_splats, SplatDecoder.PARAMS)
    raw[..., 0] = 0.30
    raw[..., 1] = -0.20
    raw[0, :, 2] = -6.0  # near
    raw[1, :, 2] = +6.0  # far

    xyz, _, _, _ = dec.activate(raw)
    projected_uv = cfg.focal * xyz[..., 0:2] / xyz[..., 2:3]
    expected_uv = dec.xy_anchor[None] + 0.38 * torch.tanh(raw[..., 0:2])

    # This is the Phase-1 invariant: depth may change, image-plane ray does not.
    assert (xyz[1, :, 2] > xyz[0, :, 2]).all()
    assert torch.allclose(projected_uv, expected_uv, atol=1e-6)
    assert torch.allclose(projected_uv[0], projected_uv[1], atol=1e-6)


def test_checkpoint_roundtrip(tmp_path: Path):
    cfg = WorldSplatConfig(image_size=24, num_splats=9, latent_dim=4, hidden_dim=32)
    model = WorldSplatVAE(cfg)
    p = tmp_path / "m.pt"
    torch.save(model.checkpoint_dict(extra={"hello": "world"}), p)
    loaded, extra = WorldSplatVAE.load_checkpoint(p)
    assert loaded.cfg == cfg
    assert extra["hello"] == "world"
    z = torch.randn(1, 4)
    with torch.no_grad():
        a = model.render_latent(z).rgb
        b = loaded.render_latent(z).rgb
    assert torch.allclose(a, b)


def test_old_checkpoint_format_is_rejected(tmp_path: Path):
    cfg = WorldSplatConfig(image_size=24, num_splats=9, latent_dim=4, hidden_dim=32)
    model = WorldSplatVAE(cfg)
    obj = model.checkpoint_dict()
    obj["format"] = "worldsplat-v0"
    p = tmp_path / "old.pt"
    torch.save(obj, p)
    with pytest.raises(ValueError, match="incompatible x/y geometry"):
        WorldSplatVAE.load_checkpoint(p)


def test_tiny_training_smoke(tmp_path: Path):
    data = tmp_path / "images"
    data.mkdir()
    for i in range(6):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:16] = (80 + i * 3, 130, 200)
        arr[16:] = (70, 100 + i * 2, 60)
        x = 3 + i * 3
        arr[10:25, x:x+6] = (180, 90 + i * 10, 60)
        Image.fromarray(arr).save(data / f"{i:03d}.png")

    out = tmp_path / "run"
    cfg = TrainConfig(
        data_dir=str(data), out_dir=str(out), image_size=24,
        num_splats=9, latent_dim=4, hidden_dim=32,
        steps=2, batch=2, preview_every=2, save_every=2, amp=False,
    )
    ckpt = train_worldsplat(cfg)
    assert ckpt.exists()
    assert (out / "preview_latest.png").exists()
