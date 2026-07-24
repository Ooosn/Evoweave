"""Topology-local motion evidence for the isolated flat-UniRig route."""

from .attention import (
    CoverageAwareMotionEvidenceCrossAttention,
    MotionEvidenceCrossAttention,
)
from .coverage import (
    PrefixSupportTargets,
    PrefixSurfaceSupportHead,
    prefix_support_distribution_loss,
    prefix_support_targets,
)
from .data import MotionEvidenceManifestDataset, motion_evidence_collate
from .encoder import (
    MotionEvidenceValues,
    TopologyMotionEncoding,
    TopologyLocalMotionEvidence,
    TopologyMotionValueEncoder,
)
from .model import (
    CoverageAwareMotionEvidenceDecoderAdapter,
    CoverageAwareTopologyMotionEvidenceUniRigAR,
    MotionEvidenceDecoderAdapter,
    MotionEvidenceMemory,
    MotionEvidenceTeacherForcingOutput,
    StaticQueryMotionEvidenceConditioner,
    TopologyMotionEvidenceUniRigAR,
)
from .supervision import (
    QuerySkinBoundaryTargets,
    query_aligned_skin_boundary_targets,
    query_aligned_skin_weights,
)

__all__ = [
    "CoverageAwareMotionEvidenceCrossAttention",
    "CoverageAwareMotionEvidenceDecoderAdapter",
    "CoverageAwareTopologyMotionEvidenceUniRigAR",
    "MotionEvidenceCrossAttention",
    "MotionEvidenceDecoderAdapter",
    "MotionEvidenceMemory",
    "MotionEvidenceManifestDataset",
    "MotionEvidenceTeacherForcingOutput",
    "MotionEvidenceValues",
    "PrefixSupportTargets",
    "PrefixSurfaceSupportHead",
    "QuerySkinBoundaryTargets",
    "StaticQueryMotionEvidenceConditioner",
    "TopologyLocalMotionEvidence",
    "TopologyMotionEncoding",
    "TopologyMotionEvidenceUniRigAR",
    "TopologyMotionValueEncoder",
    "query_aligned_skin_boundary_targets",
    "query_aligned_skin_weights",
    "motion_evidence_collate",
    "prefix_support_distribution_loss",
    "prefix_support_targets",
]
