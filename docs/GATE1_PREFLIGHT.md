# Gate 1 preflight — confident depth, weak anchor

Date: 2026-08-18

## Why this should be the first 3-D gate

Do not begin with a photorealistic scene. Begin with the smallest genuinely 3-D ambiguity: monocular projection can leave depth poorly constrained even when a learned world prior is very confident about where the point/surface probably is.

That is exactly the state this repo claims ordinary uncertainty can hide:

```text
posterior confidence      high
external geometric anchor low along depth
```

## Geometry

Let a world point be `x in R^3`. A camera observation is

```text
y = pi(T x) + noise
```

with projection Jacobian `J` around the current estimate. The measurement contributes local information

```text
Lambda_obs = J^T R^-1 J
```

A learned prior contributes

```text
Lambda_prior
```

so locally

```text
Lambda_post = Lambda_prior + sum Lambda_obs
```

The anchor state stores only accepted observation information:

```text
Lambda_anchor = sum independent observation contributions
```

For a task direction `v` (especially the poorly observed depth direction):

```text
q_ext(v) = (v^T Lambda_anchor v) / (v^T Lambda_post v)
```

A strong prior can make posterior variance small while `q_ext(depth)` remains small.

## Constructed world

Two hidden depth states project almost identically in camera A. The teacher/world prior strongly favors the common depth but the holdout world sometimes uses the rare depth.

Candidate actions:

```text
WAIT_A       another measurement from essentially the same baseline
ROUTE_SIDE   lateral camera displacement with useful parallax
ROUTE_BAD    displacement that adds little depth information
```

The experiment owns ground truth, so expected/realized information gain and recovery can be measured.

## Policies

Compare at matched observation/action cost:

```text
random
coverage heuristic
posterior entropy / variance threshold
expected posterior entropy reduction
anchor-deficit threshold
expected anchor-information gain
```

The strong attacker is expected information gain, not a dumb entropy threshold.

## Candidate finding

The project only earns something interesting if there is a regime where a strong but wrong prior suppresses ordinary posterior uncertainty enough that confidence-based exploration stops too early, while anchor-aware routing still seeks a geometrically independent view and improves holdout recovery.

Even then, do not claim active vision is new. The result would be narrower:

> **Separating source support from posterior confidence can preserve an exploration drive in directions where a learned prior is sharp but weakly grounded.**

## Mandatory controls

- weak prior: anchor-aware policy should lose its special advantage when ordinary uncertainty is already honest;
- correct strong prior: anchor-aware exploration may spend unnecessary cost, which must be reported;
- same-baseline repeats: many measurements may improve image-plane precision but should not be credited as equivalent to useful parallax in depth;
- replayed synthetic render from camera A: must not increase independent anchor;
- independent second model using only camera-A pixels: cannot count as independent geometric evidence merely because its weights differ;
- oracle independent depth sensor: positive control for genuine new depth support.

## Stop line

If a standard information-gain policy matches anchor-aware routing across wrong-prior regimes at the same cost, keep the standard policy. The anchor bookkeeping can still be useful for reporting provenance, but it has not earned a new exploration mechanism.
