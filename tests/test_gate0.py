from experiments.gate0_prediction_is_not_evidence import run


def test_gate0_registered_shape():
    r = run(n=2000)
    assert r.variance_naive_loop < r.variance_once / 20.0
    assert r.variance_anchored_loop == r.variance_once
    assert r.anchor_after_loop == r.anchor_once
    assert r.anchor_route > r.anchor_once + 0.40
    assert r.accuracy_route > r.accuracy_once + 0.20
