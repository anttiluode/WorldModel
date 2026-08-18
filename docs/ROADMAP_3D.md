# Roadmap — from anchored belief to a learned 3-D world

## Target object

A persistent world is not just a renderable 3-D hypothesis.

```text
World_t = (content_t, support_t, lineage_t)
```

- `content`: the current best 3-D scene hypothesis;
- `support`: which task-relevant directions/attributes are constrained by current or propagated observations;
- `lineage`: which original information routes support those constraints.

A generative prior may complete unseen geometry. Completion is allowed to be vivid. It is not allowed to silently relabel itself as observation.

## Why splats remain useful

The useful inheritance from the splat repos is not "Gabor waves are the world".

3-D Gaussian/surfel primitives give an explicit, local, renderable scene memory. Each primitive can carry ordinary scene state plus support metadata:

```text
position / covariance
appearance
semantic feature / primitive type
motion state
belief uncertainty
external-support score
view/source lineage
last independently observed time
```

This is compatible with existing Gaussian-SLAM / semantic-Gaussian work, which is an attacker and substrate, not a novelty claim.

## Teacher -> student path

Do not ask one giant teacher to write the final world directly.

Use teachers as modular proposal/supervision sources:

```text
RGB / short video
   |-- geometry teacher: camera + depth / point map
   |-- semantic teacher: sky / ground / wall / building / vegetation / object features
   `-- optional generative teacher: plausible completion hypotheses

                 v
          pseudo 3-D observations
                 v
       small persistent student world
```

The teacher teaches a **world vocabulary / prior**, while real views anchor a particular world instance.

Keep every teacher adapter optional so the core world state is not tied to one vendor or model.

## Minimum 3-D representation

Start with static scenes and a small number of primitive families:

```text
Gaussian/surfel       local appearance / irregular surfaces
plane                 wall / ground / road-like support
sky/background        unbounded/far-field component
```

Do not add mountains, buildings, physics, agents, weather, object permanence, or dynamics as hard-coded semantic roles in v0. Those should emerge as teacher labels or later learned categories.

## Training stages

### Stage A — teacher cache

For real image/video clips, cache teacher outputs once. The student never needs the large teacher in its inner training loop.

### Stage B — token world fitting

Fit a compact scene representation to each teacher-annotated clip. Preserve the distinction:

```text
teacher prior/proposal
real image-derived geometry constraint
independent view constraint
self-generated completion
```

### Stage C — amortized student

Train a small network to propose/update world primitives from a frame plus existing map state.

### Stage D — completion

Add a learned completion model only after the anchored mapper works. Completion predicts plausible hidden content but cannot increase `support` unless a new independently informative route arrives.

### Stage E — active ROUTE

Select views/sensors by expected support gain in task-relevant directions. Compare against entropy-only, coverage-only, and random exploration.

## Kill lines

- if a normal Gaussian-SLAM/occupancy mapper with ordinary uncertainty matches the anchor-aware system, keep the baseline and stop claiming a new object;
- if support is just a disguised confidence score, the project failed;
- if teacher hallucinations are fused as ground truth, the project failed;
- if replaying the same camera frame under a different model name increases independent support, the provenance model failed;
- if completion quality improves but source-answerability worsens, report both;
- do not call a visually complete unseen region "known".
