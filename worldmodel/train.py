from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_or_load_cache
from .worldsplat import WorldSplatConfig, WorldSplatVAE, depth_to_unit, make_preview


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "runs/worldsplat"
    depth_dir: str | None = None
    image_size: int = 64
    num_splats: int = 128
    latent_dim: int = 64
    hidden_dim: int = 384
    steps: int = 20000
    batch: int = 12
    lr: float = 3e-4
    beta_kl: float = 2e-4
    depth_weight: float = 0.8
    preview_every: int = 250
    save_every: int = 1000
    seed: int = 18018
    amp: bool = True


def _to_rgb(cache, idx, device):
    a = np.asarray(cache.rgb[idx], dtype=np.uint8).copy()
    return torch.from_numpy(a).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


def _to_depth(cache, idx, device):
    if cache.depth is None:
        return None
    a = np.asarray(cache.depth[idx], dtype=np.float32).copy()
    return torch.from_numpy(a).to(device=device, dtype=torch.float32)


def _save_preview(t: torch.Tensor, path: Path) -> None:
    from PIL import Image
    x = t.detach().clamp(0, 1).cpu()
    x = (x * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    canvas = np.concatenate(list(x), axis=0)
    Image.fromarray(canvas).save(path)


def train_worldsplat(cfg: TrainConfig, *, callback=None, stop_event=None) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def cache_cb(i, n, p):
        if callback:
            callback({"kind": "cache", "done": i, "total": n, "path": p})

    cache = build_or_load_cache(
        cfg.data_dir,
        depth_dir=cfg.depth_dir,
        image_size=cfg.image_size,
        cache_dir=out / "cache",
        progress=cache_cb,
    )
    if cache.n < 2:
        raise ValueError("need at least 2 images")

    model_cfg = WorldSplatConfig(
        image_size=cfg.image_size,
        num_splats=cfg.num_splats,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    )
    model = WorldSplatVAE(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    best = float("inf")

    for step in range(1, cfg.steps + 1):
        if stop_event is not None and stop_event.is_set():
            break
        idx = rng.integers(0, cache.n, size=cfg.batch)
        rgb = _to_rgb(cache, idx, device)
        dep = _to_depth(cache, idx, device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            render, mu, logvar, splats = model(rgb)
            l_rgb = F.l1_loss(render.rgb, rgb)
            l_coarse = F.l1_loss(F.avg_pool2d(render.rgb, 4), F.avg_pool2d(rgb, 4))
            l_kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            l_depth = rgb.new_zeros(())
            if dep is not None:
                pred_d = depth_to_unit(render.depth, model_cfg)
                w = render.coverage.detach().clamp(0.05, 1.0)
                l_depth = ((pred_d - dep).abs() * w).sum() / w.sum().clamp_min(1.0)
            xyz, sigma, color, opacity = splats
            l_sparse = opacity.mean() * 0.002
            l_center = ((xyz[..., 2] - 2.5) ** 2).mean() * (0.002 if dep is None else 0.0)
            beta = cfg.beta_kl * min(1.0, step / 2000.0)
            loss = l_rgb + 0.30 * l_coarse + beta * l_kl + (cfg.depth_weight * l_depth if dep is not None else 0.0) + l_sparse + l_center

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        val = float(loss.detach())
        best = min(best, val)
        if callback and (step == 1 or step % 25 == 0):
            callback({
                "kind": "step", "step": step, "steps": cfg.steps,
                "loss": val, "rgb": float(l_rgb.detach()), "depth": float(l_depth.detach()),
                "kl": float(l_kl.detach()), "seconds": time.time() - t0,
                "device": str(device), "has_depth": dep is not None,
            })

        if step % cfg.preview_every == 0 or step == cfg.steps:
            model.eval()
            with torch.no_grad():
                pidx = np.arange(min(4, cache.n))
                prgb = _to_rgb(cache, pidx, device)
                pmu, _ = model.encode(prgb)
                pout = model.render_latent(pmu)
                preview = make_preview(prgb, pout, model_cfg)
            p = out / "preview_latest.png"
            _save_preview(preview, p)
            if callback:
                callback({"kind": "preview", "path": str(p), "step": step})
            model.train()

        if step % cfg.save_every == 0 or step == cfg.steps:
            ckpt = model.checkpoint_dict(extra={
                "train_config": asdict(cfg),
                "depth_supervised": cache.depth is not None,
                "dataset_size": cache.n,
                "best_train_loss": best,
            })
            torch.save(ckpt, out / "worldsplat_latest.pt")

    ckpt_path = out / "worldsplat_latest.pt"
    if not ckpt_path.exists():
        torch.save(model.checkpoint_dict(extra={"train_config": asdict(cfg), "depth_supervised": cache.depth is not None, "dataset_size": cache.n}), ckpt_path)
    if callback:
        callback({"kind": "done", "path": str(ckpt_path)})
    return ckpt_path
