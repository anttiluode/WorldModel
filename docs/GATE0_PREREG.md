# Gate 0 preregistration — prediction is not evidence

Date: 2026-08-18

## Question

Can a world-state representation explicitly prevent recursive model completion from being counted as new evidence, while still allowing a genuinely independent observation route to increase support and correct a wrong prior?

This gate is intentionally not 3-D. It tests the bookkeeping invariant before geometry, rendering, or a large teacher can hide mistakes.

## State

The model carries two objects:

```text
belief          mean + posterior information
external anchor information contributed by independent observation lineages
```

For a direction `v`:

```text
q_ext(v) = (v^T Lambda_anchor v) / (v^T Lambda_post v)
```

`q_ext` is not probability of correctness. It records what fraction of local posterior precision came from accepted external evidence under this toy information model.

## Invariant

```text
prediction may propagate/change belief
prediction must not create independent external support
```

In the static Gate-0 toy, self-prediction is therefore not fused as a fresh likelihood term.

## Conditions

1. One weak sensor observation under a strong learned prior.
2. Naive self-reinjection: repeatedly fuse the current posterior as if it were a new independent observation.
3. Anchored self-reinjection: feed the same prediction back, but provenance forbids it from adding support or precision.
4. ROUTE: acquire a second independent sensor observation with distinct lineage.

## Registered checks

- naive recursive fusion reduces variance by >20x without acquiring new data;
- anchored self-reinjection leaves posterior precision unchanged in the static toy;
- anchored self-reinjection leaves external support unchanged;
- an independent route increases directional external support by >0.40;
- the independent route improves binary state accuracy by >0.20 absolute over the one-observation condition.

## Non-claims

- this is not a new Bayesian filtering theorem;
- `q_ext` is not calibrated truth probability;
- the conservative lineage rule is not a complete correlated-estimate fusion algorithm;
- the gate says nothing yet about 3-D Gaussian splats, scene completion, or active perception;
- teacher output based only on the current observation is not automatically an independent source merely because the teacher is large.

## Next gate if this passes

Construct a tiny partially observed 3-D scene in which a strong learned prior is confidently wrong about an occluded distinction. Compare:

```text
entropy/uncertainty-driven routing
vs
anchor-deficit routing
```

The interesting outcome would be a high-confidence/low-anchor state where ordinary uncertainty says "do not look" but anchor-aware routing acquires a viewpoint that resolves the distinction.
