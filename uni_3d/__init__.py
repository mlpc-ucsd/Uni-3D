# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_mask2former_config, add_uni_3d_config

# dataset loading
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.front3d_dataset_mapper import (
    Front3DDPSDatasetMapper, Front3DDatasetMapper, Front3DTestDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .uni_3d_model import Uni3D

# evaluation
from .evaluation.dvps_evaluation import Front3DDPSEvaluator, MatterportDPSEvaluator
from .evaluation.front3d_evaluation import Front3DEvaluator
