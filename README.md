# WorldModel

> **Let the model imagine the unseen world. Never let it forget which parts it imagined.**

This repo is a research instrument for a small **anchored generative world model**: a persistent scene representation that separates its current best hypothesis from the evidence that actually supports that hypothesis.

## This branch: WorldSplat + Virtual KITTI 2

`agent/worldsplat-vkitti2` is a clone of the video/demo branch `agent/worldsplat-studio`. The original Studio branch is intentionally left unchanged.

This branch exists for one specific next experiment:

> **Take the tiny rotatable WorldSplat manifold that learned a geometrically wrong hollow-face solution from RGB-only CelebA, and train the same family on an outdoor world dataset with real dense geometry supervision.**

Virtual KITTI 2 is convenient because its driving scenes provide paired RGB and 16-bit metric depth, repeated scenes under weather/time changes, stereo cameras, and camera ground truth. This v0 loader deliberately uses **Camera_0** and only the non-rotated appearance/weather variants:

```text
clone
morning
overcast
rain
fog
sunset
```

The ±15°/±30° camera-rotation variants are excluded for now because WorldSplat v0 still treats every training frame as if it came from one canonical camera. Pose-aware multi-view training is the next architecture gate.

### Easiest run: GUI

1. Download and extract the official Virtual KITTI 2 **RGB** and **depth** archives.
2. Put/extract both under one common parent folder. They may remain separate trees, for example:

```text
D:/VKITTI2/
    vkitti_2.0.3_rgb/
        Scene01/...
    vkitti_2.0.3_depth/
        Scene01/...
```

3. Switch to this branch and launch the Studio:

```bash
git fetch origin
git switch agent/worldsplat-vkitti2
pip install -r requirements-worldsplat.txt
python world_studio.py
```

4. Click **LOAD VIRTUAL KITTI 2** and choose `D:/VKITTI2` — the common parent containing both extracted trees.

The button scans and pairs RGB/depth by `(scene, variation, frame, camera)`, switches the run into Virtual KITTI 2 mode, and loads the current 12-GB-GPU overnight preset:

```text
steps       80000
image size  96
splats      256
latent      96
batch       6
camera      Camera_0
depth max   80 m
```

Then press **START TRAIN**.

### Depth handling is intentionally different here

Virtual KITTI 2 depth PNG values are centimetres. This loader converts them as:

```text
PNG integer
    -> depth metres = value * 0.01
    -> unit depth = clip(depth_metres / 80 m, 0, 1)
```

There is **no per-image percentile normalisation** in the Virtual KITTI loader. Ten metres therefore means the same training depth in every frame. This matters for the world-model experiment; the older generic relative-depth loader intentionally remains unchanged for the original Studio branch/workflow.

The cache is created locally under the chosen output directory (`runs/vkitti2/cache` by default). It is training data/cache, not something intended for GitHub.

### Command-line equivalent

```bash
python train_worldsplat.py \
  --vkitti2 "D:/VKITTI2" \
  --out runs/vkitti2 \
  --image-size 96 \
  --splats 256 \
  --latent 96 \
  --batch 6 \
  --steps 80000
```

`--vkitti2-depth-max-m` defaults to `80.0` if you want to change the global depth range.

### What this experiment can and cannot establish

This branch is still **single-frame WorldSplat v0**. Each image is encoded into its own latent scene, so correct metric depth supervision can test whether the learned splat manifold stops choosing hollow/cardboard geometry, but it is not yet one persistent world reconstructed from many camera poses.

If this run is encouraging, the next branch should use Virtual KITTI 2's camera extrinsics/intrinsics and train **one shared world state against several frames/poses**. That is the important transition from a geometry-supervised 2.5-D scene prior to an actual multi-view 3-D world model.

Official dataset/download/format page: https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/

Virtual KITTI 2 is for non-commercial use under its published dataset terms; check the official page before redistribution or commercial use.

## WorldSplat Studio — usable now

The original practical world-prior trainer/viewer lives on `agent/worldsplat-studio`.

```bash
pip install -r requirements-worldsplat.txt
python world_studio.py
```

Give it a folder of images, optionally a matching folder of relative depth maps, and train a compact latent model whose decoder emits explicit 3-D soft splats rather than pixels. The GUI can sample, encode images, interpolate worlds, and orbit the learned scene.

For the original v0 geometry-supervised mode using monocular teacher depth:

```bash
pip install transformers accelerate
python tools/make_depths.py --data D:/world_images --out D:/world_depth
python world_studio.py
```

See `WORLDSPLAT_QUICKSTART.md` and `docs/WORLDSPLAT_V0.md`.

Important: RGB-only training is an **appearance/cardboard baseline**. Monocular depth makes this a visible-surface **2.5-D** learner, not a solved full 3-D world model. True multiview consistency is the next representation gate.

## The anchored-world question

A pretrained generative model can complete an occluded scene very convincingly. If that completion is fed back into the model and treated like a new observation, confidence can grow even though no new information entered the system.

So the world state is not just:

```text
X_hat = what I think the world is
```

It is at least:

```text
World = (X_hat, support, lineage)
```

where `support` records externally anchored information and `lineage` records where it came from.

## The invariant

> **Prediction may change belief. Prediction is not new evidence.**

A camera frame may add support. A geometrically informative new view may add support. A teacher may provide a strong prior or a proposal. A self-generated completion may be useful. But replaying a prediction is not another measurement.

## Gate 0 audit

The original 1-D Gate 0 remains useful as bookkeeping/unit-test machinery, but its original 5/5 checks do **not** establish the proposed lineage/ROUTE mechanism. The current handoff records the audit: the accuracy comparison was confounded by class balance and unequal sensor precision, and scalar `q_ext` cannot exercise the directional property that matters.

The replacement gate is at least 2-D and uses actual observation geometry. A sharp learned prior can make posterior variance small in a poorly observed depth direction while directional external support remains tiny. A camera baseline shift can then add the missing geometric information.

See `HANDOFF_CURRENT.md` and `docs/GATE1_PREFLIGHT.md`.

## Where 3-D enters

The likely persistent scene substrate is an explicit map built from 3-D Gaussians/surfels plus a few structural primitives. That part is not novel: Gaussian SLAM, semantic Gaussians, persistent Gaussian memories, learned scene completion, and teacher-to-3-D distillation are active established areas.

The possible new object is narrower:

```text
generative scene content
+
direction/attribute-specific external support
+
shared-source / lineage accounting
+
active ROUTE when confidence is high but anchoring is weak
```

## Relationship to the splat repos

What is inherited:

- **SplatWorld:** a compact learned manifold can turn small coordinates into a rich structured hypothesis.
- **SplatField:** external drive and internal continuation can be continuously separated in a running generative system.
- **TheSplat5:** a rich latent belief can be kept attached to a live stream by sparse observations, while a strong prior can also project the wrong kind of world when the observation is inadequate.
- **SplatNeuron:** count resource currencies and attack compact structure with strong generic baselines.
- **SplatNeuronPlusField:** distinguish external innovation from recursive projection.

What is *not* inherited: claims about Gabors, frequency, neurons, ephaptic fields, or brain computation.

## Long target

A larger teacher should not be asked to continuously run the world. Instead it can teach/cache generic structure:

```text
many real scenes
   -> geometry / semantic teachers
   -> pseudo 3-D supervision
   -> small student learns a world vocabulary / prior
```

At deployment the small student instantiates one particular world from real observations. It can complete what it cannot see, but the support map continues to say what the current world has actually constrained.

That is the project.
