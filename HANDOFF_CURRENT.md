# WorldModel — current handoff

Date: 2026-08-18

## One-line state

> **Build a small persistent world that is free to complete missing 3-D content, but carries a separate support/provenance state so model-generated completion can never masquerade as independent observation. In parallel, WorldSplat v0 now gives us a usable trainer/viewer for learning the scene prior itself.**

## Why this repo exists

The immediate ancestor is `SplatNeuronPlusField`, specifically its separation of **external innovation** from **recursive projection**. `SplatNeuron` contributes the resource discipline: a compact observer/representation is interesting only after strong structured attackers and explicit accounting. `SplatWorld`, `SplatField`, and `TheSplat5` contribute practical representation lessons: a learned manifold can hold a rich hypothesis and sparse observations can keep it attached to a live stream, but a strong prior can also produce a plausible state after identity/source information has been lost.

Do not import old frequency, brain, or field claims here.

## Architectural hypothesis

A normal generative world model often exposes something like:

```text
X_hat_t = current best world
```

This repo starts with a stricter object:

```text
World_t = (X_hat_t, A_t, L_t)
```

where:

- `X_hat_t` is the current best scene hypothesis;
- `A_t` is external evidential support / anchoring;
- `L_t` is source lineage/provenance.

Central invariant:

> **Predictions may propagate belief. They must not manufacture evidence.**

A large teacher may install a rich prior or generate proposals. That does not mean the unseen backside of an object has been observed.

## Gate 0 audit correction

Files:

```text
docs/GATE0_PREREG.md
worldmodel/core.py
experiments/gate0_prediction_is_not_evidence.py
tests/test_core.py
tests/test_gate0.py
```

The original scalar Gate 0 is now **bookkeeping/unit-test territory**, not a scientific gate for the mechanism.

The audit found three decisive weaknesses:

- the one-observation accuracy was effectively a constant-predictor/class-balance result under the chosen prior;
- naïve self-reinjection doubles information arithmetically each pass, so the dramatic precision increase is constructed;
- the independent second route improved accuracy because it was much sharper, while changing only the lineage tag left accuracy unchanged.

Therefore do not cite the old 5/5 as evidence that lineage or independent routing helps inference.

The replacement science gate must be directional and at least 2-D. A single camera Jacobian can strongly constrain an in-plane direction while leaving depth weakly externally anchored; a sharp learned prior can make the posterior variance in both directions small while `q_ext(v)` still separates them dramatically.

That is the actual thesis to attack.

## Gate 1 / directional ROUTE

Build the smallest scene with a truly aliased view:

```text
camera A sees the same image for two hidden geometries
learned prior strongly prefers geometry 1
camera B / baseline shift can distinguish them
```

Compare ROUTE policies:

```text
random
coverage
posterior entropy / variance
expected information gain
external-anchor deficit
expected anchor gain
```

Matched-precision controls are mandatory. A second look may not win merely because it has lower sensor noise.

Lineage should not be keyed only by a source string. In 3-D, information novelty depends on source **and pose/baseline/observation geometry**. `J^T R^-1 J` is the primary geometry object; provenance is needed for shared-ancestor/correlated-estimate accounting, not as a substitute for baseline.

The candidate result is not "the generator hallucinates." That is old.

The sharp result would be:

> **A strong prior can make posterior entropy low precisely where source anchoring is weak; an anchor-aware policy can still demand a geometrically informative new view and recover a hidden distinction that confidence-only routing declines to inspect.**

## WorldSplat v0 — usable trainer/viewer

Branch:

```text
agent/worldsplat-studio
```

Files:

```text
worldmodel/worldsplat.py
worldmodel/data.py
worldmodel/train.py
train_worldsplat.py
world_studio.py
tools/make_depths.py
docs/WORLDSPLAT_V0.md
WORLDSPLAT_QUICKSTART.md
requirements-worldsplat.txt
tests/test_worldsplat.py
```

Purpose:

> **Learn the generic scene prior / vocabulary before trying to maintain one anchored online world.**

Representation:

```text
image
 -> encoder -> latent z
 -> decoder -> fixed set of 3-D soft splats
              xyz / scale / rgb / opacity
 -> perspective renderer
 -> RGB + depth
```

This is the SplatWorld architectural move lifted into a shallow scene representation: decode structured primitives and force those primitives to produce the observation.

### Data modes

```text
RGB only
    appearance-manifold / cardboard null
    depth is unidentifiable and must not be called learned geometry

RGB + cached monocular depth
    recommended v0
    teacher gives visible-surface 2.5-D supervision

true RGB-D
    stronger if available

true multiview
    next representation gate
```

`tools/make_depths.py` optionally uses a large monocular depth model offline. The teacher is not needed during subsequent WorldSplat training or viewing.

The desktop GUI can:

- select image/depth/output folders;
- train in a background thread;
- watch loss and reconstruction previews;
- load checkpoints;
- sample random latent scenes;
- encode an arbitrary image;
- morph between latent worlds;
- orbit yaw/pitch and change focal length;
- view learned RGB and depth side by side.

Local implementation validation before push:

```text
3/3 WorldSplat tests pass
renderer gradients finite
checkpoint strict round-trip passes
2-step end-to-end train/cache/preview/checkpoint smoke passes
300-step toy sky/ground/object family learns visible coarse structure
```

Do not turn the synthetic visual smoke into a scientific result.

### First real dataset advice

Do not start with arbitrary "all world" imagery. Start with a coherent family analogous to aligned faces in SplatWorld:

```text
roads / streets
suburban exteriors
mountain landscapes
rooms
building facades + surroundings
```

Then ask whether the latent manifold becomes a geometry of plausible scenes rather than a mean image.

## 3-D target after that

Use an explicit scene memory, probably 3-D Gaussians/surfels plus optional planes. Each primitive eventually carries:

```text
geometry
appearance
semantic feature
motion state
belief uncertainty
external support
lineage / shared-ancestor information
age since support
```

Teacher stack should be modular:

```text
geometry teacher -> depth / camera / point map
semantic teacher -> class / feature proposals
generative teacher -> completion prior only
```

A student is then trained to update the persistent world from a frame + previous world. The expensive teacher can be cached offline.

## Missing dynamics

The current anchored Gaussian core still lacks:

- process/time update;
- support aging;
- correlated-estimate fusion;
- foundation-model-as-shared-ancestor accounting.

The correlated case may be the most scientifically interesting provenance problem: if one teacher prior influences many later estimates, naïve fusion can repeatedly reuse common information. Existing common-information / covariance-intersection ideas are mandatory attackers before any novelty claim.

## Scientific attackers

- ordinary Bayesian/uncertainty-aware occupancy mapping;
- Gaussian-SLAM / 3-D Gaussian persistent maps;
- semantic Gaussian fusion with reliability weighting;
- coverage-driven active perception;
- entropy/information-gain active perception;
- teacher-student 3-D distillation;
- generative completion with calibrated uncertainty;
- correlated-estimate/common-information fusion and covariance intersection;
- conventional image VAE against WorldSplat's explicit 3-D representation.

The repo only earns a distinct contribution if source support/provenance or explicit scene structure buys something those baselines do not.

## Stop lines

- no "world model without hallucination" claim;
- no claim that splats are novel 3-D memory;
- no claim that teacher knowledge is evidence about the current token world;
- no old Gate-0 5/5 headline after the audit;
- no source-string novelty used as a substitute for 3-D baseline geometry;
- no double counting predictions or repeated correlated observations;
- no claim that RGB-only WorldSplat learned real depth;
- no claim that monocular pseudo-depth gives unseen backside truth;
- no brain/ephaptic claims in this repo;
- no inference from green CI or a pretty GUI.
