import pyvene as pv

from .pyvene_compat import register_pyvene_model_compat

EMBED_LAYER = -1
EMBED_COMPONENT = "model.embed_tokens"
BLOCK_COMPONENT = "block_output"


def intervention_component(layer: int) -> str:
    """Return the pyvene component string for a training/eval layer index."""
    return EMBED_COMPONENT if layer == EMBED_LAYER else BLOCK_COMPONENT


def build_reft_representation(layer: int, low_rank_dim: int, intervention) -> dict:
    """Build a single pyvene representation dict for ReftConfig."""
    rep = {
        "low_rank_dimension": low_rank_dim,
        "intervention": intervention,
    }
    if layer == EMBED_LAYER:
        rep["component"] = EMBED_COMPONENT
    else:
        rep["layer"] = layer
        rep["component"] = BLOCK_COMPONENT
    return rep


class ReftModel(pv.IntervenableModel):
    def __init__(self, config, model, **kwargs):
        super().__init__(config, model, **kwargs)

    def print_trainable_parameters(self):
        trainable_intervention_parameters = 0
        linked_keys = set()

        for key, value in self.interventions.items():
            intervention_obj = value[0] if isinstance(value, (list, tuple)) else value
            if isinstance(intervention_obj, pv.TrainableIntervention):
                if key in self._intervention_reverse_link:
                    reverse_key = self._intervention_reverse_link[key]
                    if reverse_key not in linked_keys:
                        linked_keys.add(reverse_key)
                        trainable_intervention_parameters += sum(
                            p.numel()
                            for p in intervention_obj.parameters()
                            if p.requires_grad
                        )
                else:
                    trainable_intervention_parameters += sum(
                        p.numel()
                        for p in intervention_obj.parameters()
                        if p.requires_grad
                    )

        trainable_model_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        all_model_parameters = sum(p.numel() for p in self.model.parameters())
        total_trainable = trainable_intervention_parameters + trainable_model_parameters

        print(
            f"trainable intervention params: {trainable_intervention_parameters:,d} || "
            f"trainable model params: {trainable_model_parameters:,d}\n"
            f"model params: {all_model_parameters:,d} || "
            f"trainable%: {100 * total_trainable / all_model_parameters}"
        )


def get_reft_model(model, reft_config, set_device=True):
    register_pyvene_model_compat()
    reft_model = ReftModel(reft_config, model)
    if set_device:
        reft_model.set_device(model.device)
    reft_model.disable_model_gradients()
    return reft_model
