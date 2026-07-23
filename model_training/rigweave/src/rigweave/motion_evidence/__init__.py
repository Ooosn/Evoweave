"""Topology-local motion evidence for the isolated flat-UniRig route."""

from .attention import MotionEvidenceCrossAttention
from .data import MotionEvidenceManifestDataset, motion_evidence_collate
from .encoder import (
    MotionEvidenceValues,
    TopologyMotionEncoding,
    TopologyLocalMotionEvidence,
    TopologyMotionValueEncoder,
)
from .model import (
    MotionEvidenceDecoderAdapter,
    MotionEvidenceMemory,
    MotionEvidenceTeacherForcingOutput,
    StaticQueryMotionEvidenceConditioner,
    TopologyMotionEvidenceUniRigAR,
)
from .supervision import QuerySkinBoundaryTargets, query_aligned_skin_boundary_targets

__all__ = [
    "MotionEvidenceCrossAttention",
    "MotionEvidenceDecoderAdapter",
    "MotionEvidenceMemory",
    "MotionEvidenceManifestDataset",
    "MotionEvidenceTeacherForcingOutput",
    "MotionEvidenceValues",
    "QuerySkinBoundaryTargets",
    "StaticQueryMotionEvidenceConditioner",
    "TopologyLocalMotionEvidence",
    "TopologyMotionEncoding",
    "TopologyMotionEvidenceUniRigAR",
    "TopologyMotionValueEncoder",
    "query_aligned_skin_boundary_targets",
    "motion_evidence_collate",
]
