import numpy as np

from worldmodel import AnchoredGaussian, Evidence, NaiveGaussian, SourceKind


def test_self_prediction_does_not_change_anchored_belief():
    b = AnchoredGaussian.from_prior([0.0], 2.0)
    b.fuse(Evidence.isotropic([1.0], 3.0, kind=SourceKind.SENSOR, source_id="cam"))
    before_mean = b.mean.copy()
    before_info = b.info.copy()
    before_anchor = b.anchor_info.copy()

    for _ in range(5):
        assert not b.fuse(b.predict_static())

    np.testing.assert_allclose(b.mean, before_mean)
    np.testing.assert_allclose(b.info, before_info)
    np.testing.assert_allclose(b.anchor_info, before_anchor)


def test_external_route_increases_anchor():
    b = AnchoredGaussian.from_prior([0.0], 2.0)
    assert b.fuse(Evidence.isotropic([1.0], 3.0, kind=SourceKind.SENSOR, source_id="cam-a"))
    q1 = b.directional_anchor([1.0])
    assert b.fuse(Evidence.isotropic([1.1], 3.0, kind=SourceKind.SENSOR, source_id="cam-b"))
    q2 = b.directional_anchor([1.0])
    assert q2 > q1


def test_teacher_prior_does_not_increment_external_anchor():
    b = AnchoredGaussian.from_prior([0.0], 1.0)
    b.fuse(Evidence.isotropic([2.0], 4.0, kind=SourceKind.TEACHER_PRIOR, source_id="teacher"))
    assert b.directional_anchor([1.0]) == 0.0
    assert b.mean[0] > 1.0


def test_naive_self_reinjection_becomes_overconfident():
    n = NaiveGaussian.from_prior([0.0], 1.0)
    n.fuse(Evidence.isotropic([1.0], 1.0, kind=SourceKind.SENSOR, source_id="cam"))
    v0 = n.directional_variance([1.0])
    for _ in range(6):
        n.self_reinject()
    assert n.directional_variance([1.0]) < v0 / 20.0
