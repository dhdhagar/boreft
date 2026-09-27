import warnings

import pyvene as pv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer

from boreft.pyreft.losses import (
    LinearAnnealMap,
    clamp_logvar,
    effective_loss_coeffs,
    token_greedy_margin_loss,
)
from boreft.pyreft.sdpo import SDPOConfig, compute_sdpo_loss


class ReftTrainer(Trainer):
    def __init__(
        self,
        *args,
        word_sim_matrix=None,
        neighborhood_block_size=None,
        kl_beta: float = 0.0,
        steps_per_epoch: int = 1,
        lambda_ce: float = 1.0,
        use_margin_loss: bool = False,
        margin_loss_margin: float = 0.1,
        lambda_sdpo: float = 0.0,
        linear_annealing_map: LinearAnnealMap | None = None,
        sdpo_cfg: SDPOConfig | None = None,
        sdpo_base_model=None,
        sdpo_student_prompt_ids=None,
        sdpo_teacher_prompt_ids=None,
        sdpo_target_ids=None,
        **kwargs,
    ):
        if "tokenizer" in kwargs:
            import transformers as _tv

            major, minor = (int(x) for x in _tv.__version__.split(".")[:2])
            if (major, minor) >= (4, 47):
                kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.word_sim_matrix = word_sim_matrix
        self.neighborhood_block_size = neighborhood_block_size
        self.kl_beta = kl_beta
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.lambda_ce = lambda_ce
        self.use_margin_loss = use_margin_loss
        self.margin_loss_margin = margin_loss_margin
        self.lambda_sdpo = lambda_sdpo
        self.linear_annealing_map: LinearAnnealMap = dict(linear_annealing_map or {})
        self.sdpo_cfg = sdpo_cfg
        self.sdpo_base_model = sdpo_base_model
        self.sdpo_student_prompt_ids = sdpo_student_prompt_ids
        self.sdpo_teacher_prompt_ids = sdpo_teacher_prompt_ids
        self.sdpo_target_ids = sdpo_target_ids
        # Lazy off-policy teacher-sample cache: word_id -> list of sampled token-id
        # sequences. Sampled once per word (teacher distribution is fixed) and reused.
        self._sdpo_offpolicy_cache: dict = {}

    def _current_loss_coeffs(self) -> dict[str, float]:
        return effective_loss_coeffs(
            {
                "kl_beta": float(self.kl_beta),
                "lambda_ce": float(self.lambda_ce),
                "lambda_sdpo": float(self.lambda_sdpo),
            },
            self.linear_annealing_map,
            global_step=int(self.state.global_step),
            steps_per_epoch=self.steps_per_epoch,
        )

    def _current_kl_beta(self) -> float:
        return float(self._current_loss_coeffs()["kl_beta"])

    @staticmethod
    def _set_sdpo_step_cache(intervenable, *, active: bool) -> None:
        for v in intervenable.interventions.values():
            iv = v[0] if isinstance(v, (list, tuple)) else v
            if hasattr(iv, "_shared_b"):
                iv._shared_b = {} if active else None
            if hasattr(iv, "_shared_bias"):
                iv._shared_bias = {} if active else None

    def compute_loss(
        self, intervenable: pv.IntervenableModel, inputs, return_outputs=False, **kwargs
    ):
        # Share this step's reparameterized bias (VAE) and bias-network rows across
        # the CE and SDPO forwards so both objectives backprop through a single
        # reparameterization / encoder+MLP node. The stores are set before the CE
        # forward populates them and ALWAYS cleared afterward (try/finally) so they
        # can never leak into later (eval-time) stochastic forwards; the cached
        # tensors stay referenced by the autograd graph for the pending backward.
        # _shared_b and _shared_bias must be distinct dicts (different value types).
        coeffs = self._current_loss_coeffs()
        sdpo_ready = (
            self.lambda_sdpo > 0.0
            and self.sdpo_cfg is not None
            and self.sdpo_base_model is not None
            and "subspaces" in inputs
            and intervenable.model.training
        )
        sdpo_active = sdpo_ready and float(coeffs["lambda_sdpo"]) > 0.0
        if sdpo_active:
            self._set_sdpo_step_cache(intervenable, active=True)
        try:
            return self._compute_loss_inner(
                intervenable, inputs, return_outputs, coeffs=coeffs
            )
        finally:
            if sdpo_active:
                self._set_sdpo_step_cache(intervenable, active=False)

    def _compute_loss_inner(self, intervenable, inputs, return_outputs, *, coeffs):
        unit_locations = None
        if "intervention_locations" in inputs:
            if inputs["intervention_locations"].dim() == 3:
                unit_locations = {
                    "sources->base": (
                        None,
                        inputs["intervention_locations"].permute(1, 0, 2).tolist(),
                    )
                }
            else:
                unit_locations = {"sources->base": (None, 0)}

        base_outputs, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist()
            if "subspaces" in inputs
            else None,
        )

        def _get_intervention(v):
            return v[0] if isinstance(v, (list, tuple)) else v

        output = cf_outputs if cf_outputs is not None else base_outputs

        def _weighted_ce_from_logits(logits, labels, weights):
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token_loss = loss_fct(
                shift_logits.view(-1, logits.shape[-1]),
                shift_labels.view(-1),
            ).view(shift_labels.shape)
            mask = (shift_labels != -100).float()
            per_sample_loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(
                min=1
            )
            return (per_sample_loss * weights).sum() / weights.sum()

        # Margin loss always needs raw logits. CE uses HuggingFace output.loss unless
        # neighborhood augmentation supplies per-example weights.
        if self.use_margin_loss:
            if output.logits is None:
                raise RuntimeError("Margin loss requires logits from intervenable forward")
            weights = inputs.get("weights")
            lm_loss = token_greedy_margin_loss(
                output.logits,
                inputs["labels"],
                margin=self.margin_loss_margin,
                sample_weights=weights,
            )
        elif "weights" in inputs and output.logits is not None:
            weights = inputs["weights"].to(output.logits.device)  # [batch]
            lm_loss = _weighted_ce_from_logits(
                output.logits, inputs["labels"], weights
            )
        elif output.loss is not None:
            lm_loss = output.loss
        elif output.logits is not None:
            weights = torch.ones(
                inputs["labels"].shape[0],
                device=output.logits.device,
                dtype=torch.float32,
            )
            lm_loss = _weighted_ce_from_logits(
                output.logits, inputs["labels"], weights
            )
        else:
            raise RuntimeError("Intervenable forward returned neither loss nor logits")

        aux_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)
        kl_beta_now = float(coeffs["kl_beta"])
        lambda_ce_now = float(coeffs["lambda_ce"])
        lambda_sdpo_now = float(coeffs["lambda_sdpo"])
        mu_norm_stat = None
        bias_var_stat = None
        for intervention_val in intervenable.interventions.values():
            iv = _get_intervention(intervention_val)
            if hasattr(iv, "kl_beta"):
                iv.kl_beta = kl_beta_now
            if hasattr(iv, "auxiliary_loss"):
                aux_loss = aux_loss + iv.auxiliary_loss().to(device=lm_loss.device)
            # VAE bias diagnostics: avg L2 norm of mu and avg learned variance over
            # the current batch (mu/logvar are only set for the distributional bias).
            last_mu = getattr(iv, "last_mu", None)
            if last_mu is not None:
                mu_norm_stat = float(last_mu.detach().float().norm(dim=-1).mean())
                last_logvar = getattr(iv, "last_logvar", None)
                if last_logvar is not None:
                    bias_var_stat = float(
                        clamp_logvar(last_logvar.detach().float()).exp().mean()
                    )
        total_loss = lambda_ce_now * lm_loss + aux_loss

        sdpo_loss_stat = None
        sdpo_metrics = None
        if (
            lambda_sdpo_now > 0.0
            and self.lambda_sdpo > 0.0
            and self.sdpo_cfg is not None
            and self.sdpo_base_model is not None
            and "subspaces" in inputs
            and intervenable.model.training
        ):
            word_ids = inputs["subspaces"].reshape(inputs["subspaces"].shape[0], -1)[
                :, 0
            ].tolist()
            tokenizer = (
                getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            )
            sdpo_loss, sdpo_metrics = compute_sdpo_loss(
                intervenable=intervenable,
                base_model=self.sdpo_base_model,
                tokenizer=tokenizer,
                word_ids=word_ids,
                student_prompt_ids=self.sdpo_student_prompt_ids,
                teacher_prompt_ids=self.sdpo_teacher_prompt_ids,
                target_ids=self.sdpo_target_ids,
                cfg=self.sdpo_cfg,
                collator=self.data_collator,
                device=lm_loss.device,
                offpolicy_cache=self._sdpo_offpolicy_cache,
                draw_seed=int(getattr(self.args, "seed", 42)) * 1_000_003
                + int(self.state.global_step),
            )
            sdpo_loss = sdpo_loss.to(device=lm_loss.device, dtype=total_loss.dtype)
            total_loss = total_loss + lambda_sdpo_now * sdpo_loss
            sdpo_loss_stat = float(sdpo_loss.detach())

        lm_loss_key = "margin_loss" if self.use_margin_loss else "ce_loss"
        log_dict = {
            lm_loss_key: float(lm_loss.detach()),
            "aux_loss": float(aux_loss.detach()),
            "loss": float(total_loss.detach()),
        }
        if sdpo_loss_stat is not None:
            log_dict["sdpo_loss"] = sdpo_loss_stat
            log_dict["sdpo_n_seq"] = sdpo_metrics["sdpo_n_seq"]
            log_dict["sdpo_n_tokens"] = sdpo_metrics["sdpo_n_tokens"]
        if "kl_beta" in self.linear_annealing_map or self.kl_beta > 0:
            log_dict["kl_beta_effective"] = kl_beta_now
        if "lambda_ce" in self.linear_annealing_map:
            log_dict["lambda_ce_effective"] = lambda_ce_now
        if "lambda_sdpo" in self.linear_annealing_map:
            log_dict["lambda_sdpo_effective"] = lambda_sdpo_now
        if mu_norm_stat is not None:
            log_dict["mu_norm"] = mu_norm_stat
        if bias_var_stat is not None:
            log_dict["bias_var"] = bias_var_stat
        self.log(log_dict)

        return (output, output) if return_outputs else total_loss


class ReftTrainerForCausalLM(ReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        from boreft.data_utils import NeighborhoodBlockBatchSampler

        nb = getattr(self, "neighborhood_block_size", None)
        if nb is not None and nb > 0:
            ds = self.train_dataset
            n = len(ds)
            if n % nb != 0:
                raise ValueError(
                    f"Train dataset length {n} must be divisible by neighborhood_block_size {nb}"
                )
            bs = self._train_batch_size
            if bs % nb != 0:
                bs_old = bs
                bs = max(nb, ((bs + nb - 1) // nb) * nb)
                self.args.per_device_train_batch_size = bs
                self._train_batch_size = bs
                warnings.warn(
                    f"per_device_train_batch_size={bs_old} rounded up to {bs} "
                    f"(multiple of neighborhood block_size={nb}).",
                    stacklevel=2,
                )
            blocks_per_batch = bs // nb
            num_blocks = n // nb
            if blocks_per_batch > num_blocks:
                bs_old = bs
                blocks_per_batch = num_blocks
                bs = blocks_per_batch * nb
                self.args.per_device_train_batch_size = bs
                self._train_batch_size = bs
                warnings.warn(
                    f"per_device_train_batch_size={bs_old} capped down to {bs} "
                    f"({blocks_per_batch} anchor blocks × block_size={nb}).",
                    stacklevel=2,
                )
            batch_sampler = NeighborhoodBlockBatchSampler(
                dataset_len=n,
                block_size=nb,
                blocks_per_batch=blocks_per_batch,
                shuffle=True,
                seed=int(getattr(self.args, "seed", 42)),
                drop_last=True,
            )
            return DataLoader(
                ds,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
        )
