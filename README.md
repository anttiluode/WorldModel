# WorldModel

> **Let the model imagine the unseen world. Never let it forget which parts it imagined.**

This repo is a research instrument for a small **anchored generative world model**: a persistent scene representation that separates its current best hypothesis from the evidence that actually supports that hypothesis.

## This branch: Phase 1 ray-coordinate fix

`agent/worldsplat-vkitti2-rayfix` is a strict A/B child of `agent/worldsplat-vkitti2`. The completed 80k VKITTI2 branch is intentionally left intact as the control.

### Paper / technical note

**[WorldSplat is not 3D Gaussian Splatting — From single-scene reconstruction to a learned scene manifold](docs/WORLDSPLAT_VS_3DGS.md)** explains the central distinction from ordinary scene-specific 3DGS, the SplatWorld / TinyAvatar / SlapStack lineage that led here, why CelebA was deceptively friendly, what the failed VKITTI2 run exposed, and why paired stereo views are the next important world-model test.

The first VKITTI2 run learned visible road/tree/sky structure but retained a starved, blurry peripheral field. The code audit found a concrete coordinate conflict: the decoder inherited bounded image-like x/y anchors from the SplatWorld/TinyAvatar lineage, then treated them as world-space x/y and perspective-divided them by learned depth.

Old parameterisation:

```text
x,y = bounded anchor + offset
z   = learned depth
u   = focal*x/z
v   = focal*y/z
```

As z increased, a splat lost access to the image periphery. Metric depth therefore forced far buildings/sky inward while RGB reconstruction asked them to remain at the frame edge.

**Phase 1 changes only the coordinate semantics and the anchor-grid asymmetry:**

```text
u,v = bounded image-plane ray anchor + offset
z   = learned depth
x,y = (u,v) * z / focal
```

At zero camera rotation, projection now gives `focal*x/z == u` and `focal*y/z == v` exactly, so image position and depth are independent. This is a geometry correction, **not** stereo, SLAM, persistence, object binding, or a world-layer claim.

The old `_anchor_grid(512)` also made a 23x23 lattice and returned the first 512 entries, truncating one end of the grid. This branch selects 512 points evenly over the complete lattice so its centroid and full x/y extent remain symmetric.

Old `worldsplat-v0` checkpoints are deliberately rejected on this branch because their x/y outputs have different semantics. New checkpoints use format `worldsplat-v0-rayfix`.

### Strict retrain A/B

For the cleanest comparison with the completed run, use the same settings that produced it:

```text
steps       80000
image size  128
splats      512
latent      96
batch       6
camera      Camera_0
depth max   80 m
```

The GUI defaults to those values on this branch and writes to `runs/vkitti2_rayfix` so it does not overwrite the old `runs/vkitti2` control.

Judge the model first by `preview_latest.png` or **ENCODE IMAGE** on a real VKITTI frame. `NEW RANDOM WORLD` samples `z ~ N(0,I)` and remains a separate test of whether the weak-KL VAE prior matches the aggregate encoder posterior.

The Phase-1 falsifier is narrow: **does decoupling ray position from depth reduce the peripheral splat fog while preserving useful metric-depth organization?** If not, do not rescue the result by silently adding more capacity.

## Virtual KITTI 2 loader

Virtual KITTI 2 is convenient because its driving scenes provide paired RGB and 16-bit metric depth, repeated scenes under weather/time changes, stereo cameras, and camera ground truth. This v0 loader deliberately uses **Camera_0** and only the non-rotated appearance/weather variants:

```text
clone
morning
overcast
rain
fog
sunset
```

The ±15°/±30° camera-rotation variants are excluded for now because WorldSplat still treats every training frame as a separate latent scene. Pose-aware multi-view training is the next architecture gate.

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
git switch --track origin/agent/worldsplat-vkitti2-rayfix
pip install -r requirements-worldsplat.txt
python world_studio.py
```

4. Click **LOAD VIRTUAL KITTI 2** and choose `D:/VKITTI2` — the common parent containing both extracted trees.

Then press **START TRAIN**.

### Depth handling is intentionally fixed-scale

Virtual KITTI 2 depth PNG values are centimetres. This loader converts them as:

```text
PNG integer
    -> depth metres = value * 0.01
    -> unit depth = clip(depth_metres / 80 m, 0, 1)
```

There is **no per-image percentile normalisation** in the Virtual KITTI loader. Ten metres therefore means the same training depth in every frame.

### Command-line equivalent

```bash
python train_worldsplat.py \
  --vkitti2 "D:/VKITTI2" \
  --out runs/vkitti2_rayfix \
  --image-size 128 \
  --splats 512 \
  --latent 96 \
  --batch 6 \
  --steps 80000
```

`--vkitti2-depth-max-m` defaults to `80.0`.

### What this experiment can and cannot establish

This branch is still **single-frame WorldSplat**. Each image is encoded into its own latent scene. Correct metric-depth supervision can test the splat geometry, but it is not yet one persistent world reconstructed from many camera poses.

The dataset's two cameras are deliberately *not* consumed jointly yet. A later stereo branch should pair Camera_0 and Camera_1 at the same frame and force **one decoded scene** to render both views. That can provide multiview geometric pressure even without using the supplied depth maps, but merely mixing both camera folders as unrelated training images would not do that.

## WorldSplat Studio lineage

The original practical world-prior trainer/viewer lives on `agent/worldsplat-studio`. The first VKITTI2 adaptation lives on `agent/worldsplat-vkitti2`; this branch changes only the Phase-1 geometry described above.

Important: RGB-only training is an **appearance/cardboard baseline**. Single-frame metric depth makes this a geometry-supervised visible-surface learner, not a solved full 3-D world model. True multiview consistency is the next representation gate.

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
