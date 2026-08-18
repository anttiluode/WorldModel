from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class WorldSplatConfig:
    image_size: int = 64
    num_splats: int = 128
    latent_dim: int = 64
    hidden_dim: int = 384
    z_near: float = 1.0
    z_far: float = 4.0
    focal: float = 1.25
    background: float = 0.08


@dataclass
class RenderOutput:
    rgb: torch.Tensor
    depth: torch.Tensor
    coverage: torch.Tensor


def _anchor_grid(n: int) -> torch.Tensor:
    side = int(n**0.5)
    if side * side < n:
        side += 1
    ys = torch.linspace(-0.85, 0.85, side)
    xs = torch.linspace(-0.85, 0.85, side)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    return pts[:n]


class SoftSplatRenderer(nn.Module):
    """Small differentiable 3-D soft-splat renderer.

    This is deliberately not a production 3D Gaussian Splatting renderer. It
    uses isotropic 3-D splats, perspective projection, depth sorting and
    front-to-back alpha compositing. That keeps the entire training/viewing
    instrument pure PyTorch and easy to inspect.
    """

    def __init__(self, cfg: WorldSplatConfig):
        super().__init__()
        self.cfg = cfg
        coords = torch.linspace(-1.0, 1.0, cfg.image_size)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", gx[None, None].contiguous().clone())
        self.register_buffer("grid_y", gy[None, None].contiguous().clone())

    @staticmethod
    def _rotation(yaw_deg: float, pitch_deg: float, device, dtype) -> torch.Tensor:
        yaw = torch.as_tensor(yaw_deg * 3.141592653589793 / 180.0, device=device, dtype=dtype)
        pitch = torch.as_tensor(pitch_deg * 3.141592653589793 / 180.0, device=device, dtype=dtype)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        one = torch.ones_like(cy)
        zero = torch.zeros_like(cy)
        ry = torch.stack([
            torch.stack([cy, zero, sy]),
            torch.stack([zero, one, zero]),
            torch.stack([-sy, zero, cy]),
        ])
        rx = torch.stack([
            torch.stack([one, zero, zero]),
            torch.stack([zero, cp, -sp]),
            torch.stack([zero, sp, cp]),
        ])
        return rx @ ry

    def forward(
        self,
        xyz: torch.Tensor,
        sigma: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
        *,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        focal: float | None = None,
    ) -> RenderOutput:
        b, n, _ = xyz.shape
        focal = float(self.cfg.focal if focal is None else focal)
        center = xyz.new_tensor([0.0, 0.0, (self.cfg.z_near + self.cfg.z_far) * 0.5])
        rot = self._rotation(yaw_deg, pitch_deg, xyz.device, xyz.dtype)
        p = (xyz - center) @ rot.T + center

        # Stable front-to-back order. Sorting itself is piecewise constant, but
        # the projected positions/depths remain differentiable within an order.
        order = torch.argsort(p[..., 2], dim=1)
        gather3 = order[..., None].expand(-1, -1, 3)
        p = torch.gather(p, 1, gather3)
        color = torch.gather(color, 1, gather3)
        sigma = torch.gather(sigma, 1, order)
        opacity = torch.gather(opacity, 1, order)

        z = p[..., 2].clamp_min(0.15)
        u = focal * p[..., 0] / z
        v = focal * p[..., 1] / z
        sig2d = (focal * sigma / z).clamp(0.008, 0.45)

        dx = self.grid_x - u[..., None, None]
        dy = self.grid_y - v[..., None, None]
        gauss = torch.exp(-0.5 * (dx.square() + dy.square()) / sig2d[..., None, None].square())
        alpha = 1.0 - torch.exp(-3.0 * opacity[..., None, None] * gauss)
        alpha = alpha.clamp(0.0, 0.995)

        one = torch.ones((b, 1, self.cfg.image_size, self.cfg.image_size), device=xyz.device, dtype=xyz.dtype)
        trans = torch.cumprod(torch.cat([one, (1.0 - alpha + 1e-6)], dim=1), dim=1)[:, :-1]
        contrib = trans * alpha
        mass = contrib.sum(dim=1).clamp(0.0, 1.0)

        rgb = (contrib[:, :, None] * color[:, :, :, None, None]).sum(dim=1)
        bg = float(self.cfg.background)
        rgb = rgb + (1.0 - mass[:, None]) * bg

        depth_num = (contrib * z[..., None, None]).sum(dim=1)
        depth = depth_num / contrib.sum(dim=1).clamp_min(1e-6)
        far = xyz.new_full(depth.shape, self.cfg.z_far)
        depth = torch.where(mass > 1e-4, depth, far)
        return RenderOutput(rgb=rgb.clamp(0.0, 1.0), depth=depth, coverage=mass)


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        chans = [3, 32, 64, 128, 192]
        layers: list[nn.Module] = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mu = nn.Linear(chans[-1], latent_dim)
        self.logvar = nn.Linear(chans[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 6.0)


class SplatDecoder(nn.Module):
    PARAMS = 8  # dx,dy,z,sigma,r,g,b,opacity

    def __init__(self, cfg: WorldSplatConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, cfg.num_splats * self.PARAMS),
        )
        self.register_buffer("xy_anchor", _anchor_grid(cfg.num_splats))

    def activate(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = raw.view(raw.shape[0], self.cfg.num_splats, self.PARAMS)
        xy = self.xy_anchor[None] + 0.38 * torch.tanh(raw[..., 0:2])
        z = self.cfg.z_near + (self.cfg.z_far - self.cfg.z_near) * torch.sigmoid(raw[..., 2:3])
        xyz = torch.cat([xy, z], dim=-1)
        sigma = 0.025 + 0.20 * torch.sigmoid(raw[..., 3])
        color = torch.sigmoid(raw[..., 4:7])
        opacity = torch.sigmoid(raw[..., 7])
        return xyz, sigma, color, opacity

    def forward(self, z: torch.Tensor):
        return self.activate(self.net(z))


class WorldSplatVAE(nn.Module):
    def __init__(self, cfg: WorldSplatConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.latent_dim)
        self.decoder = SplatDecoder(cfg)
        self.renderer = SoftSplatRenderer(cfg)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_splats(self, z: torch.Tensor):
        return self.decoder(z)

    def render_latent(self, z: torch.Tensor, *, yaw_deg: float = 0.0, pitch_deg: float = 0.0, focal: float | None = None) -> RenderOutput:
        xyz, sigma, color, opacity = self.decode_splats(z)
        return self.renderer(xyz, sigma, color, opacity, yaw_deg=yaw_deg, pitch_deg=pitch_deg, focal=focal)

    def forward(self, x: torch.Tensor) -> tuple[RenderOutput, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        splats = self.decode_splats(z)
        out = self.renderer(*splats)
        return out, mu, logvar, splats

    def checkpoint_dict(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format": "worldsplat-v0",
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> tuple["WorldSplatVAE", dict[str, Any]]:
        obj = torch.load(path, map_location=device, weights_only=False)
        if obj.get("format") != "worldsplat-v0":
            raise ValueError("not a WorldSplat v0 checkpoint")
        cfg = WorldSplatConfig(**obj["config"])
        model = cls(cfg).to(device)
        model.load_state_dict(obj["state_dict"], strict=True)
        model.eval()
        return model, obj.get("extra", {})


def depth_to_unit(depth: torch.Tensor, cfg: WorldSplatConfig) -> torch.Tensor:
    return ((depth - cfg.z_near) / max(cfg.z_far - cfg.z_near, 1e-6)).clamp(0.0, 1.0)


def make_preview(input_rgb: torch.Tensor, render: RenderOutput, cfg: WorldSplatConfig) -> torch.Tensor:
    """Return Bx3xHx(3W) preview: input | reconstruction | depth."""
    d = depth_to_unit(render.depth, cfg)
    d3 = d[:, None].repeat(1, 3, 1, 1)
    return torch.cat([input_rgb, render.rgb, d3], dim=-1)
