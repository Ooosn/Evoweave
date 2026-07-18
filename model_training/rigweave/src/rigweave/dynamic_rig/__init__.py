"""Dynamic mesh sequence to autoregressive rig generation modules."""

from .motion_encoder import (
    AnchorWiseAlternatingMotionEncoder,
    FrameTypeAnchorWiseAlternatingMotionEncoder,
    LegacyTemporalMotionEncoder,
    TemporalMotionEncoder,
)
from .model import DynamicRigConditioner
from .sampling import TrackableSurfaceReferences, TrackableSurfaceSamples, sample_trackable_surface
from .surface_tokenizer import FixedQuerySurfaceTokenizer
from .joint_count_sampler import JointCountMixtureSampler, parse_joint_count_bin_uppers
from .topology_sampler import (
    TopologyFamilyMixtureSampler,
    load_parent_topology_signatures,
)
from .unirig_wrapper import DynamicRigUniRigAR
from .puppeteer_dynamic import (
    PuppeteerDynamicRigDataset,
    PuppeteerDynamicRigModel,
    PuppeteerJointTokenizer,
    import_puppeteer_decoder,
    load_puppeteer_decoder_state,
    load_puppeteer_target_aware_pos_embed,
    puppeteer_dynamic_collate,
)

__all__ = [
    "DynamicRigConditioner",
    "DynamicRigUniRigAR",
    "FixedQuerySurfaceTokenizer",
    "AnchorWiseAlternatingMotionEncoder",
    "FrameTypeAnchorWiseAlternatingMotionEncoder",
    "LegacyTemporalMotionEncoder",
    "TemporalMotionEncoder",
    "TrackableSurfaceReferences",
    "TrackableSurfaceSamples",
    "sample_trackable_surface",
    "JointCountMixtureSampler",
    "parse_joint_count_bin_uppers",
    "TopologyFamilyMixtureSampler",
    "load_parent_topology_signatures",
    "PuppeteerDynamicRigDataset",
    "PuppeteerDynamicRigModel",
    "PuppeteerJointTokenizer",
    "import_puppeteer_decoder",
    "load_puppeteer_decoder_state",
    "load_puppeteer_target_aware_pos_embed",
    "puppeteer_dynamic_collate",
]
