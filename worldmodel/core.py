from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable

import numpy as np


class SourceKind(str, Enum):
    """Where a constraint on the world came from."""

    SENSOR = "sensor"
    INDEPENDENT_MODEL = "independent_model"
    TEACHER_PRIOR = "teacher_prior"
    SELF_PREDICTION = "self_prediction"


_EXTERNAL_KINDS = {SourceKind.SENSOR, SourceKind.INDEPENDENT_MODEL}


@dataclass(frozen=True)
class Evidence:
    """A Gaussian constraint with explicit source lineage.

    `lineage` names the original information sources that ultimately support this
    constraint. Replaying a derived estimate with the same lineage must not be
    counted as a new independent observation.
    """

    mean: np.ndarray
    precision: np.ndarray
    kind: SourceKind
    lineage: FrozenSet[str]

    @staticmethod
    def isotropic(
        mean: Iterable[float],
        precision: float,
        *,
        kind: SourceKind,
        source_id: str,
    ) -> "Evidence":
        mean_arr = np.asarray(tuple(mean), dtype=np.float64)
        if precision < 0:
            raise ValueError("precision must be non-negative")
        return Evidence(
            mean=mean_arr,
            precision=np.eye(mean_arr.size, dtype=np.float64) * float(precision),
            kind=kind,
            lineage=frozenset({source_id}),
        )


@dataclass
class AnchoredGaussian:
    """Gaussian world hypothesis plus separately tracked external support.

    The posterior information matrix describes *belief*. `anchor_info` describes
    how much independent, externally sourced measurement information has directly
    constrained the state. Teacher priors and self-predictions can shape belief,
    but do not increment external anchoring.

    This first implementation is intentionally conservative: any evidence whose
    lineage overlaps previously consumed external lineage is not allowed to add
    external information again. That is stricter than a full correlated-estimate
    fusion model, but it makes the no-double-counting invariant explicit.
    """

    mean: np.ndarray
    info: np.ndarray
    anchor_info: np.ndarray
    external_lineage: set[str] = field(default_factory=set)

    @classmethod
    def from_prior(
        cls,
        mean: Iterable[float],
        precision: float | np.ndarray,
    ) -> "AnchoredGaussian":
        mean_arr = np.asarray(tuple(mean), dtype=np.float64)
        if np.isscalar(precision):
            info = np.eye(mean_arr.size, dtype=np.float64) * float(precision)
        else:
            info = np.asarray(precision, dtype=np.float64)
        if info.shape != (mean_arr.size, mean_arr.size):
            raise ValueError("prior precision shape does not match mean")
        return cls(
            mean=mean_arr.copy(),
            info=info.copy(),
            anchor_info=np.zeros_like(info),
        )

    @property
    def covariance(self) -> np.ndarray:
        return np.linalg.pinv(self.info)

    def fuse(self, evidence: Evidence) -> bool:
        """Fuse evidence into belief; return whether external anchor increased."""
        if evidence.mean.shape != self.mean.shape:
            raise ValueError("evidence dimension mismatch")
        if evidence.precision.shape != self.info.shape:
            raise ValueError("evidence precision shape mismatch")

        # A self prediction is a continuation of the current belief, not an
        # independent likelihood term. In this static Gate-0 world, fusing it
        # again would be pure double counting.
        if evidence.kind == SourceKind.SELF_PREDICTION:
            return False

        old_eta = self.info @ self.mean
        new_info = self.info + evidence.precision
        new_eta = old_eta + evidence.precision @ evidence.mean
        self.info = new_info
        self.mean = np.linalg.pinv(new_info) @ new_eta

        increased_anchor = False
        if evidence.kind in _EXTERNAL_KINDS:
            novel = evidence.lineage.isdisjoint(self.external_lineage)
            if novel:
                self.anchor_info = self.anchor_info + evidence.precision
                self.external_lineage.update(evidence.lineage)
                increased_anchor = True
        return increased_anchor

    def predict_static(self, *, source_id: str = "self") -> Evidence:
        """Return a prediction derived only from this belief."""
        return Evidence(
            mean=self.mean.copy(),
            precision=self.info.copy(),
            kind=SourceKind.SELF_PREDICTION,
            lineage=frozenset(self.external_lineage | {source_id}),
        )

    def directional_anchor(self, direction: Iterable[float]) -> float:
        """Posterior precision fraction along `direction` due to external support."""
        v = np.asarray(tuple(direction), dtype=np.float64)
        if v.shape != self.mean.shape:
            raise ValueError("direction dimension mismatch")
        denom = float(v @ self.info @ v)
        if denom <= 0:
            return 0.0
        numer = float(v @ self.anchor_info @ v)
        return float(np.clip(numer / denom, 0.0, 1.0))

    def directional_variance(self, direction: Iterable[float]) -> float:
        v = np.asarray(tuple(direction), dtype=np.float64)
        if v.shape != self.mean.shape:
            raise ValueError("direction dimension mismatch")
        return float(v @ self.covariance @ v)


@dataclass
class NaiveGaussian:
    """Reference filter that treats every incoming estimate as independent evidence."""

    mean: np.ndarray
    info: np.ndarray

    @classmethod
    def from_prior(cls, mean: Iterable[float], precision: float) -> "NaiveGaussian":
        mean_arr = np.asarray(tuple(mean), dtype=np.float64)
        return cls(mean_arr, np.eye(mean_arr.size, dtype=np.float64) * float(precision))

    def fuse(self, evidence: Evidence) -> None:
        old_eta = self.info @ self.mean
        self.info = self.info + evidence.precision
        self.mean = np.linalg.pinv(self.info) @ (old_eta + evidence.precision @ evidence.mean)

    def self_reinject(self) -> None:
        pseudo = Evidence(
            mean=self.mean.copy(),
            precision=self.info.copy(),
            kind=SourceKind.SELF_PREDICTION,
            lineage=frozenset({"self"}),
        )
        self.fuse(pseudo)

    def directional_variance(self, direction: Iterable[float]) -> float:
        v = np.asarray(tuple(direction), dtype=np.float64)
        cov = np.linalg.pinv(self.info)
        return float(v @ cov @ v)
