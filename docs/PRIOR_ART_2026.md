# Prior-art boundary — 2026-08-18

This note exists to stop future iterations from rediscovering occupied territory and calling it the result.

## 3-D geometry teachers already exist

**VGGT: Visual Geometry Grounded Transformer** (arXiv:2503.11651) directly infers camera parameters, point maps, depth maps and point tracks from one, a few, or many views. A large feed-forward geometry teacher is therefore a practical component, not a novelty claim.

## Gaussian worlds / memories already exist

**GaussianWorld** (arXiv:2412.10373) uses an explicit Gaussian world model for streaming 3-D occupancy and scene evolution.

**Gaussian Splatting SLAM** (arXiv:2312.06741) and related Gaussian-SLAM systems already use 3-D Gaussians as online persistent mapping/tracking representations.

**GSMem** (arXiv:2603.19137) explicitly uses 3-D Gaussian Splatting as persistent spatial memory and renders novel views for embodied reasoning.

Therefore:

```text
3-D Gaussians as persistent memory
```

is occupied territory.

## Teacher semantics -> 3-D is occupied

**Semantic Gaussians** (arXiv:2403.15624) projects knowledge/features from pretrained 2-D models into 3-D Gaussians for open-vocabulary scene understanding.

**Splat and Distill** (arXiv:2602.06032) lifts teacher features into an explicit 3-D Gaussian representation and uses novel-view feature maps for 3-D-aware student distillation.

Therefore:

```text
large 2-D teacher teaches a smaller / 3-D-aware representation
```

is also occupied territory.

## Uncertainty/reliability-aware Gaussian fusion is occupied

**CG-SLAM** (arXiv:2403.16095) uses uncertainty-aware 3-D Gaussian SLAM.

**VCS-SLAM** (arXiv:2606.29494) explicitly validates semantic evidence geometrically and suppresses unreliable/occluded semantic updates rather than fusing all 2-D semantic priors uniformly.

Therefore:

```text
attach confidence/reliability to Gaussian-map updates
```

is not enough.

## World-model hallucination / coverage is active work

**Hallucination in World Models is Predictable and Preventable** (arXiv:2606.27326) reports visually fluent world-model rollouts drifting from true dynamics, relates failures to low-coverage state-action regions, and uses failure predictors to guide targeted real data collection.

That is very close to the motivation here. Do not claim that "world models need grounding/coverage" is new.

## Candidate narrow gap for this repo

The current hypothesis is more specific:

```text
scene content / posterior belief
           !=
independent external support for that content
```

and source lineage is retained so repeated predictions or correlated descendants of the same observation do not silently become additional evidence.

Then active sensing can react to:

```text
high confidence + weak external anchor
```

rather than only high uncertainty.

I did not identify an exact match for this whole package in the quick 2026 pass above. That is not a novelty proof. Factor graphs, Bayesian filtering, covariance intersection/common-information fusion, active vision, SLAM, evidential reasoning and uncertainty calibration are huge literatures and must be searched more deeply before any claim.

## Practical consequence

The first contribution to chase is not better rendering. It is a controlled result where explicit support/provenance changes a decision or failure mode relative to strong uncertainty/information-gain baselines.
