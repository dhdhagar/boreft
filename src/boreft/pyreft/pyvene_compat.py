"""Register pyvene intervention hook mappings for models missing upstream support."""

_registered = False


def register_pyvene_model_compat() -> None:
    """Register hook mappings for models not yet in upstream pyvene (e.g. Gemma3)."""
    global _registered
    if _registered:
        return

    from pyvene.models.gemma2.modelings_intervenable_gemma2 import (
        gemma2_lm_type_to_dimension_mapping,
        gemma2_lm_type_to_module_mapping,
        gemma2_type_to_dimension_mapping,
        gemma2_type_to_module_mapping,
    )
    from pyvene.models.intervenable_modelcard import (
        type_to_dimension_mapping,
        type_to_module_mapping,
    )

    try:
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3ForCausalLM,
            Gemma3TextModel,
        )
    except ImportError:
        _registered = True
        return

    for model_cls, mod_map, dim_map in (
        (
            Gemma3ForCausalLM,
            gemma2_lm_type_to_module_mapping,
            gemma2_lm_type_to_dimension_mapping,
        ),
        (
            Gemma3TextModel,
            gemma2_type_to_module_mapping,
            gemma2_type_to_dimension_mapping,
        ),
    ):
        if model_cls not in type_to_module_mapping:
            type_to_module_mapping[model_cls] = mod_map
        if model_cls not in type_to_dimension_mapping:
            type_to_dimension_mapping[model_cls] = dim_map

    _registered = True
