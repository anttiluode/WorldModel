# WorldModel — current handoff

Date: 2026-08-18

## One-line state

> **Build a small persistent world that is free to complete missing 3-D content, but carries a separate support/provenance state so model-generated completion can never masquerade as independent observation.**

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

## Gate 0

Files:

```text
docs/GATE0_PREREG.md
worldmodel/core.py
experiments/gate0_prediction_is_not_evidence.py
tests/test_core.py
tests/test_gate0.py
```

Gate 0 is deliberately a scalar/vector Gaussian information toy. It must establish the bookkeeping before we add 3-D rendering.

## Gate 1 if Gate 0 passes

Build the smallest scene with a truly aliased view:

```text
camera A sees the same image for two hidden geometries
learned prior strongly prefers geometry 1
camera B can distinguish them
```

Compare ROUTE policies:

```text
random
coverage
posterior entropy
external-anchor deficit
```

The candidate result is not "the generator hallucinates." That is old.

The sharp result would be:

> **A strong prior can make posterior entropy low precisely where source anchoring is weak; an anchor-aware policy can still demand a new view and recover a hidden distinction that confidence-only routing declines to inspect.**

## 3-D target after that

Use an explicit scene memory, probably 3-D Gaussians/surfels plus optional planes. Each primitive eventually carries:

```text
geometry
appearance
semantic feature
motion state
belief uncertainty
external support
lineage / independent-view IDs
age since support
```

Teacher stack should be modular:

```text
geometry teacher -> depth / camera / point map
semantic teacher -> class / feature proposals
generative teacher -> completion prior only
```

A student is then trained to update the persistent world from a frame + previous world. The expensive teacher can be cached offline.

## Scientific attackers

- ordinary Bayesian/uncertainty-aware occupancy mapping;
- Gaussian-SLAM / 3-D Gaussian persistent maps;
- semantic Gaussian fusion with reliability weighting;
- coverage-driven active perception;
- entropy/information-gain active perception;
- teacher-student 3-D distillation;
- generative completion with calibrated uncertainty.

The repo only earns a distinct contribution if source support/provenance buys something those baselines do not.

## Stop lines

- no "world model without hallucination" claim;
- no claim that splats are novel 3-D memory;
- no claim that teacher knowledge is evidence about the current token world;
- no double counting predictions or repeated observations with shared lineage;
- no 3-D beauty demo before Gate 1's ambiguity instrument works;
- no brain/ephaptic claims in this repo;
- no inference from green CI.
