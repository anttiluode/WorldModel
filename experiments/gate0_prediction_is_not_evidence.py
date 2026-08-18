"""Gate 0: prediction is not evidence.

A strong but imperfect prior makes a binary world state look certain after one
weak measurement. We compare naive self-reinjection, anchored self-reinjection,
and a genuinely independent second observation route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldmodel import AnchoredGaussian, Evidence, NaiveGaussian, SourceKind


@dataclass
class Result:
    accuracy_once: float
    accuracy_naive_loop: float
    accuracy_anchored_loop: float
    accuracy_route: float
    variance_once: float
    variance_naive_loop: float
    variance_anchored_loop: float
    variance_route: float
    anchor_once: float
    anchor_after_loop: float
    anchor_route: float


def sign_accuracy(estimates: np.ndarray, truth: np.ndarray) -> float:
    pred = np.where(estimates >= 0.0, 1.0, -1.0)
    return float(np.mean(pred == truth))


def run(seed: int = 18018, n: int = 4000, self_steps: int = 6) -> Result:
    rng = np.random.default_rng(seed)
    truth = rng.choice(np.array([-1.0, 1.0]), size=n)

    # Useful but biased learned prior: common-world knowledge, not token truth.
    prior_mean = 0.65
    prior_precision = 6.0

    first_noise = 1.5
    second_noise = 0.35
    first_precision = 1.0 / first_noise**2
    second_precision = 1.0 / second_noise**2

    once, naive_loop, anchored_loop, routed = [], [], [], []
    var_once, var_naive, var_anchor, var_route = [], [], [], []
    q_once, q_loop, q_route = [], [], []

    for idx, z in enumerate(truth):
        y1 = z + rng.normal(0.0, first_noise)
        first = Evidence.isotropic(
            [y1], first_precision, kind=SourceKind.SENSOR, source_id=f"cam-a:{idx}"
        )

        anchored = AnchoredGaussian.from_prior([prior_mean], prior_precision)
        anchored.fuse(first)
        once.append(anchored.mean[0])
        var_once.append(anchored.directional_variance([1.0]))
        q_once.append(anchored.directional_anchor([1.0]))

        naive = NaiveGaussian.from_prior([prior_mean], prior_precision)
        naive.fuse(first)
        for _ in range(self_steps):
            naive.self_reinject()
        naive_loop.append(naive.mean[0])
        var_naive.append(naive.directional_variance([1.0]))

        for _ in range(self_steps):
            anchored.fuse(anchored.predict_static())
        anchored_loop.append(anchored.mean[0])
        var_anchor.append(anchored.directional_variance([1.0]))
        q_loop.append(anchored.directional_anchor([1.0]))

        # ROUTE: distinct source, independent noise, genuinely new constraint.
        y2 = z + rng.normal(0.0, second_noise)
        second = Evidence.isotropic(
            [y2], second_precision, kind=SourceKind.SENSOR, source_id=f"cam-b:{idx}"
        )
        anchored.fuse(second)
        routed.append(anchored.mean[0])
        var_route.append(anchored.directional_variance([1.0]))
        q_route.append(anchored.directional_anchor([1.0]))

    once_a = np.asarray(once)
    naive_a = np.asarray(naive_loop)
    anchored_a = np.asarray(anchored_loop)
    routed_a = np.asarray(routed)

    return Result(
        accuracy_once=sign_accuracy(once_a, truth),
        accuracy_naive_loop=sign_accuracy(naive_a, truth),
        accuracy_anchored_loop=sign_accuracy(anchored_a, truth),
        accuracy_route=sign_accuracy(routed_a, truth),
        variance_once=float(np.mean(var_once)),
        variance_naive_loop=float(np.mean(var_naive)),
        variance_anchored_loop=float(np.mean(var_anchor)),
        variance_route=float(np.mean(var_route)),
        anchor_once=float(np.mean(q_once)),
        anchor_after_loop=float(np.mean(q_loop)),
        anchor_route=float(np.mean(q_route)),
    )


def main() -> None:
    r = run()
    print("Gate 0 — prediction is not evidence")
    print("-----------------------------------")
    print(f"accuracy once          {r.accuracy_once:.4f}")
    print(f"accuracy naive loop    {r.accuracy_naive_loop:.4f}")
    print(f"accuracy anchored loop {r.accuracy_anchored_loop:.4f}")
    print(f"accuracy ROUTE         {r.accuracy_route:.4f}")
    print()
    print(f"variance once          {r.variance_once:.6f}")
    print(f"variance naive loop    {r.variance_naive_loop:.6f}")
    print(f"variance anchored loop {r.variance_anchored_loop:.6f}")
    print(f"variance ROUTE         {r.variance_route:.6f}")
    print()
    print(f"external anchor once   {r.anchor_once:.4f}")
    print(f"external anchor loop   {r.anchor_after_loop:.4f}")
    print(f"external anchor ROUTE  {r.anchor_route:.4f}")

    confidence_inflation = r.variance_naive_loop < r.variance_once / 20.0
    no_anchor_inflation = abs(r.anchor_after_loop - r.anchor_once) < 1e-12
    anchored_static = abs(r.variance_anchored_loop - r.variance_once) < 1e-12
    route_adds_support = r.anchor_route > r.anchor_once + 0.40
    route_improves = r.accuracy_route > r.accuracy_once + 0.20

    print()
    print("registered checks")
    print(f"SELF_CONFIDENCE_INFLATES_WITHOUT_NEW_DATA  {confidence_inflation}")
    print(f"ANCHORED_SELF_LOOP_ADDS_NO_SUPPORT          {no_anchor_inflation}")
    print(f"ANCHORED_SELF_LOOP_ADDS_NO_PRECISION        {anchored_static}")
    print(f"INDEPENDENT_ROUTE_ADDS_SUPPORT              {route_adds_support}")
    print(f"INDEPENDENT_ROUTE_IMPROVES_ACCURACY          {route_improves}")

    if not all(
        [confidence_inflation, no_anchor_inflation, anchored_static, route_adds_support, route_improves]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
