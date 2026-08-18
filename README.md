# WorldModel

> **Let the model imagine the unseen world. Never let it forget which parts it imagined.**

This repo is a research instrument for a small **anchored generative world model**: a persistent scene representation that separates its current best hypothesis from the evidence that actually supports that hypothesis.

## WorldSplat Studio — usable now

The first practical world-prior trainer/viewer lives on `agent/worldsplat-studio`.

```bash
pip install -r requirements-worldsplat.txt
python world_studio.py
```

Give it a folder of images, optionally a matching folder of relative depth maps, and train a compact latent model whose decoder emits explicit 3-D soft splats rather than pixels. The GUI can sample, encode images, interpolate worlds, and orbit the learned scene.

For the recommended v0 geometry-supervised mode:

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
