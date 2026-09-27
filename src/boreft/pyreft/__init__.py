import pyvene as pv

from .pyvene_compat import register_pyvene_model_compat

register_pyvene_model_compat()

from .bias_network import TargetBiasNetwork
from .semantic_encoder import (
    DEFAULT_LORA_TARGETS,
    ENCODER_POOLING_CHOICES,
    POOLING_INSTRUCTION_MEAN,
    POOLING_LAST_INSTRUCTION,
    SemanticEncoder,
    build_semantic_encoder,
    capture_penultimate,
    encode_definition_inputs,
    normalize_encoder_pooling,
    pooling_requires_instruction_mask,
    pooling_uses_instruction_mask,
)
from .interventions import DistributionalWordIntervention
from .interventions import LoreftPerWordBiasIntervention

ReftConfig = pv.IntervenableConfig
from .utils import (
    EMBED_LAYER,
    build_reft_representation,
    get_reft_model,
    intervention_component,
)
from .reft_trainer import ReftTrainer, ReftTrainerForCausalLM

__all__ = [
    "ReftConfig",
    "DistributionalWordIntervention",
    "LoreftPerWordBiasIntervention",
    "TargetBiasNetwork",
    "SemanticEncoder",
    "build_semantic_encoder",
    "capture_penultimate",
    "encode_definition_inputs",
    "normalize_encoder_pooling",
    "pooling_uses_instruction_mask",
    "pooling_requires_instruction_mask",
    "ENCODER_POOLING_CHOICES",
    "POOLING_LAST_INSTRUCTION",
    "POOLING_INSTRUCTION_MEAN",
    "DEFAULT_LORA_TARGETS",
    "EMBED_LAYER",
    "build_reft_representation",
    "get_reft_model",
    "intervention_component",
    "ReftTrainer",
    "ReftTrainerForCausalLM",
]
