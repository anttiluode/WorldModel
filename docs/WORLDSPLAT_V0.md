# WorldSplat v0 — trainer/viewer instrument

## Purpose

WorldSplat v0 is the first usable world-prior instrument in this repo.

It intentionally separates two problems:

1. **world prior learning** — what do scenes tend to look like?
2. **anchored online world state** — which parts of one particular scene are actually supported by current observations?

This file implements (1). The older `worldmodel/core.py` line is about (2). Do not silently merge their claims.

## Representation

Each image is encoded to a latent `z`. The decoder emits a fixed set of soft 3-D splats:

```text
z
 -> N splats
    xyz
    scale
    rgb
    opacity
 -> perspective soft-splat renderer
 -> RGB + rendered depth
```

`x,y` begin from a shared image-plane anchor grid with learned offsets. `z`, scale, colour and opacity are learned per sample through the decoder.

The renderer is a small inspectable PyTorch implementation with depth sorting and front-to-back alpha compositing. It is **not** a production 3DGS rasterizer and should not be benchmarked as one.

## What the training data means

### RGB only

This is a deliberate null / visual baseline.

A single RGB view does not determine depth. The model can learn an appearance manifold and a depth convention that helps reconstruction, but the resulting `z` geometry is not externally anchored. Orbiting it is useful specifically because it exposes whether the learned scene is merely a billboard/cardboard arrangement.

### RGB + relative depth

Recommended v0 mode.

Provide matching `.npy` or `.png` depth maps with:

```text
0 = near
1 = far
```

The helper `tools/make_depths.py` can precompute relative depth from a monocular depth teacher. This makes the teacher an **offline geometry labeler**. The small WorldSplat model trains afterward without the teacher in the loop.

Relative monocular depth still does not make unseen backsides true. It gives the decoder a shallow geometric target for the visible scene.

### True RGB-D / multiview

Better. If true depth is available, normalize it consistently and feed it through the same interface.

True multiview scene training is the next representation gate because it can directly penalize wrong novel views. v0 does not pretend single-view reconstruction solved that problem.

## Run

### Optional: precompute depth

```bash
pip install transformers accelerate
python tools/make_depths.py --data D:/world_images --out D:/world_depth
```

Inspect the depth PNGs. WorldSplat expects `0=near, 1=far`. `--no-invert` exists if the chosen teacher already emits that convention.

### Train from the command line

```bash
python train_worldsplat.py \
  --data D:/world_images \
  --depth-dir D:/world_depth \
  --out runs/world1 \
  --image-size 64 \
  --splats 128 \
  --latent 64 \
  --batch 12 \
  --steps 20000
```

Outputs:

```text
runs/world1/worldsplat_latest.pt
runs/world1/preview_latest.png
runs/world1/cache/...
```

The preview is:

```text
INPUT | RECONSTRUCTION | LEARNED DEPTH
```

### GUI

```bash
python world_studio.py
```

The studio can:

- choose an image folder;
- choose an optional depth folder;
- train in a background thread;
- watch loss and reconstruction previews;
- load any `worldsplat_latest.pt`;
- sample a random latent world;
- encode an arbitrary image into the learned scene manifold;
- interpolate between latent worlds;
- change yaw / pitch / focal length live;
- display belief render and learned depth side by side.

## Suggested first datasets

Do **not** begin with every kind of image on Earth. SplatWorld learned one constrained family (aligned faces); start with a similarly coherent world family so we can see whether a scene manifold forms.

Useful first folders might be one of:

```text
roads / streets
Finnish suburban exteriors
mountain landscapes
rooms
building facades + surroundings
```

A few thousand varied images are more informative than 200 images, but v0 is designed so a few hundred are enough to reveal whether the representation is learning anything at all.

## What to watch for

Good signs:

- reconstructions stop behaving like flat colour averages;
- depth has stable large-scale structure when depth-supervised;
- latent interpolation changes scene structure continuously rather than crossfading two whole images;
- small camera orbit gives coherent parallax for visible geometry;
- random samples remain scene-like rather than collapsing to a single mean scene.

Kill signs:

- random samples are one mean scene;
- orbit immediately tears the image into a billboard;
- depth is constant or unrelated to teacher depth;
- model simply assigns one splat per image patch and learns no reusable scene organization;
- a conventional image VAE of similar size gives equal usefulness with much less training/render cost.

## Relation to SplatWorld

The useful inheritance is not Gabor frequency mythology. It is the architectural discipline:

> **decode a compact latent into explicit structured primitives, then make those primitives produce the observation.**

SplatWorld used localized 2-D Gabor packets. WorldSplat v0 uses localized 3-D soft splats.

If the explicit 3-D object buys nothing under depth/multiview tests, keep that negative result and move to a stronger representation rather than protecting the splat story.
