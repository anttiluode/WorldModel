from __future__ import annotations

import argparse

from worldmodel.train import TrainConfig, train_worldsplat


def main() -> None:
    ap = argparse.ArgumentParser(description="Train WorldSplat v0 from a folder of images.")
    ap.add_argument("--data", required=True, help="image folder (recursive)")
    ap.add_argument("--depth-dir", default=None, help="optional matching .npy/.png relative-depth maps (0 near, 1 far)")
    ap.add_argument("--out", default="runs/worldsplat")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--splats", type=int, default=128)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta-kl", type=float, default=2e-4)
    ap.add_argument("--depth-weight", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=18018)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data,
        depth_dir=args.depth_dir,
        out_dir=args.out,
        image_size=args.image_size,
        num_splats=args.splats,
        latent_dim=args.latent,
        hidden_dim=args.hidden,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        beta_kl=args.beta_kl,
        depth_weight=args.depth_weight,
        seed=args.seed,
        amp=not args.no_amp,
    )

    def cb(m):
        if m["kind"] == "cache":
            print(f"cache {m['done']}/{m['total']}: {m['path']}")
        elif m["kind"] == "step":
            print(f"step {m['step']:6d}/{m['steps']} loss={m['loss']:.5f} rgb={m['rgb']:.5f} depth={m['depth']:.5f} kl={m['kl']:.5f} {m['device']}")
        elif m["kind"] == "preview":
            print(f"preview -> {m['path']}")
        elif m["kind"] == "done":
            print(f"checkpoint -> {m['path']}")

    train_worldsplat(cfg, callback=cb)


if __name__ == "__main__":
    main()
