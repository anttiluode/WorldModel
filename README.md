# WorldModel

> **Let the model imagine the unseen world. Never let it forget which parts it imagined.**

This repo is a research instrument for a small **anchored generative world model**: a persistent scene representation that separates its current best hypothesis from the evidence that actually supports that hypothesis.

The project begins from a simple failure mode. A pretrained generative model can complete an occluded scene very convincingly. If that completion is fed back into the model and treated like a new observation, confidence can grow even though no new information entered the system.

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

A camera frame may add support. A genuinely independent view may add support. A teacher may provide a strong prior or a proposal. A self-generated completion may be useful. But replaying a prediction is not another measurement.

## Gate 0 — deliberately tiny

Before 3-D, splats, or a large teacher, the repo tests the bookkeeping in a one-dimensional ambiguous world:

```text
strong learned prior
      +
weak real observation
      v
confident-ish belief
      |
      | WAIT: self-predict / self-reinject
      v
naive system becomes extremely confident
without becoming more externally supported

      | ROUTE: independent second observation
      v
support rises and the hidden state becomes easier to recover
```

Run:

```bash
python -m pip install -e . pytest
python -m pytest -q
python experiments/gate0_prediction_is_not_evidence.py
```

## Where 3-D enters

The likely scene substrate is an explicit persistent map built from 3-D Gaussians/surfels plus a few structural primitives. That part is not novel: Gaussian SLAM, semantic Gaussians, persistent Gaussian memories, learned scene completion, and teacher-to-3-D distillation are active established areas.

The possible new object is narrower:

```text
generative scene content
+
direction/attribute-specific external support
+
source lineage
+
active ROUTE when confidence is high but anchoring is weak
```

See `docs/ROADMAP_3D.md` and `HANDOFF_CURRENT.md`.

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
