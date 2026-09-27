import unittest

import torch

from boreft.data_utils import IGNORE_INDEX, build_reft_row
from boreft.intervention_marker import intervention_position_list
from boreft.pyreft import DistributionalWordIntervention
from boreft.pyreft.losses import sdpo_distillation_loss
from boreft.pyreft.sdpo import (
    SDPOConfig,
    _dedupe_records,
    _draw_offpolicy,
    _ensure_offpolicy_cached,
    _gather_next_token_logits,
    _left_pad,
    _trim_generated,
    compute_sdpo_loss,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class _EvalAware:
    """Minimal nn.Module-like train/eval interface for test doubles."""

    training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


class _FakeOut:
    def __init__(self, logits):
        self.logits = logits


def _fake_collator(rows):
    """Minimal stand-in for ReftDataCollator (right-pad + stack)."""
    pad = _FakeTokenizer.pad_token_id
    R = len(rows)
    max_len = max(len(r["input_ids"]) for r in rows)
    input_ids = torch.full((R, max_len), pad, dtype=torch.long)
    labels = torch.full((R, max_len), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((R, max_len), dtype=torch.long)
    for i, r in enumerate(rows):
        ln = len(r["input_ids"])
        input_ids[i, :ln] = r["input_ids"]
        labels[i, :ln] = r["labels"]
        attn[i, :ln] = r["attention_mask"]
    intloc = torch.tensor([r["intervention_locations"] for r in rows], dtype=torch.long)
    subspaces = torch.stack([r["subspaces"] for r in rows])
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "intervention_locations": intloc,
        "subspaces": subspaces,
    }


class _FakeIntervenable(_EvalAware):
    """Returns logits = fixed_noise * scale, so grad flows into ``scale``."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def __call__(self, inputs, unit_locations=None, labels=None, subspaces=None):
        R, T = inputs["input_ids"].shape
        g = torch.Generator().manual_seed(0)
        noise = torch.randn(R, T, self.vocab, generator=g)
        return None, _FakeOut(noise * self.scale)

    def generate(self, *a, **k):  # pragma: no cover
        raise AssertionError("generation should not run in the gold-only test")


class _FakeBase(_EvalAware):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, input_ids=None, attention_mask=None):
        R, T = input_ids.shape
        g = torch.Generator().manual_seed(1)
        return _FakeOut(torch.randn(R, T, self.vocab, generator=g))

    def generate(self, *a, **k):  # pragma: no cover
        raise AssertionError("generation should not run in the gold-only test")


class TestSDPODistillationLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.s = torch.randn(7, 32)
        self.t = torch.randn(7, 32)

    def test_zero_when_identical(self):
        logits = torch.randn(5, 16)
        for div in ("forward_kl", "reverse_kl", "js"):
            val = sdpo_distillation_loss(logits, logits.clone(), divergence=div)
            self.assertAlmostEqual(float(val), 0.0, places=5, msg=div)

    def test_nonnegative(self):
        for div in ("forward_kl", "reverse_kl", "js"):
            self.assertGreaterEqual(
                float(sdpo_distillation_loss(self.s, self.t, divergence=div)) + 1e-6,
                0.0,
                msg=div,
            )

    def test_forward_kl_matches_reference(self):
        """forward_kl == mean_token KL(teacher || student) at T=1."""
        s_logp = torch.log_softmax(self.s, dim=-1)
        t_logp = torch.log_softmax(self.t, dim=-1)
        ref = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
        got = sdpo_distillation_loss(self.s, self.t, divergence="forward_kl")
        self.assertTrue(torch.allclose(got, ref, atol=1e-6))

    def test_reverse_kl_matches_reference(self):
        s_logp = torch.log_softmax(self.s, dim=-1)
        t_logp = torch.log_softmax(self.t, dim=-1)
        ref = (s_logp.exp() * (s_logp - t_logp)).sum(-1).mean()
        got = sdpo_distillation_loss(self.s, self.t, divergence="reverse_kl")
        self.assertTrue(torch.allclose(got, ref, atol=1e-6))

    def test_teacher_is_stopgrad(self):
        """Gradients flow into the student logits only; teacher is detached."""
        s = self.s.clone().requires_grad_(True)
        t = self.t.clone().requires_grad_(True)
        sdpo_distillation_loss(s, t, divergence="forward_kl").backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(torch.any(s.grad != 0))
        self.assertIsNone(t.grad)

    def test_temperature_squared_scaling(self):
        """At T, the loss is T**2 * KL of the temperature-softened distributions."""
        T = 2.0
        s_logp = torch.log_softmax(self.s / T, dim=-1)
        t_logp = torch.log_softmax(self.t / T, dim=-1)
        ref = (T**2) * (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
        got = sdpo_distillation_loss(
            self.s, self.t, temperature=T, divergence="forward_kl"
        )
        self.assertTrue(torch.allclose(got, ref, atol=1e-6))

    def test_js_symmetric(self):
        a = sdpo_distillation_loss(self.s, self.t, divergence="js")
        b = sdpo_distillation_loss(self.t, self.s, divergence="js")
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_empty_is_differentiable_zero(self):
        s = torch.zeros(0, 16, requires_grad=True)
        loss = sdpo_distillation_loss(s, torch.zeros(0, 16))
        self.assertEqual(float(loss), 0.0)
        loss.backward()  # must not raise

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            sdpo_distillation_loss(self.s, self.t, divergence="bogus")
        with self.assertRaises(ValueError):
            sdpo_distillation_loss(self.s, self.t, temperature=0.0)
        with self.assertRaises(ValueError):
            sdpo_distillation_loss(self.s, torch.randn(3, 32))
        with self.assertRaises(ValueError):
            sdpo_distillation_loss(self.s, self.t, topk=-1)

    def test_topk_zero_when_identical(self):
        logits = torch.randn(5, 16)
        val = sdpo_distillation_loss(
            logits, logits.clone(), divergence="reverse_kl", topk=4
        )
        self.assertAlmostEqual(float(val), 0.0, places=5)

    def test_topk_full_vocab_matches_dense(self):
        dense = sdpo_distillation_loss(self.s, self.t, divergence="reverse_kl")
        via_k = sdpo_distillation_loss(
            self.s, self.t, divergence="reverse_kl", topk=self.s.shape[-1]
        )
        via_zero = sdpo_distillation_loss(
            self.s, self.t, divergence="reverse_kl", topk=0
        )
        self.assertTrue(torch.allclose(dense, via_k, atol=1e-5))
        self.assertTrue(torch.allclose(dense, via_zero, atol=1e-5))

    def test_topk_reverse_kl_matches_sdpo_a3(self):
        k = 5
        s_logp = torch.log_softmax(self.s, dim=-1)
        t_logp = torch.log_softmax(self.t, dim=-1)
        _, indices = torch.topk(self.s, k, dim=-1)
        s_sel = s_logp.gather(-1, indices)
        t_sel = t_logp.gather(-1, indices)
        s_p = s_sel.exp()
        t_p = t_sel.exp()
        head = (s_p * (s_sel - t_sel)).sum(dim=-1)
        s_tail = (1.0 - s_p.sum(dim=-1)).clamp(min=0.0)
        t_tail = (1.0 - t_p.sum(dim=-1)).clamp(min=0.0)
        tail = s_tail * (
            torch.log(s_tail.clamp(min=1e-8)) - torch.log(t_tail.clamp(min=1e-8))
        )
        ref = (head + tail).mean()
        got = sdpo_distillation_loss(
            self.s, self.t, divergence="reverse_kl", topk=k, add_tail=True
        )
        self.assertTrue(torch.allclose(got, ref, atol=1e-5))

    def test_topk_teacher_is_stopgrad(self):
        s = self.s.clone().requires_grad_(True)
        t = self.t.clone().requires_grad_(True)
        sdpo_distillation_loss(s, t, divergence="reverse_kl", topk=4).backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(torch.any(s.grad != 0))
        self.assertIsNone(t.grad)

    def test_topk_js_is_finite(self):
        val = sdpo_distillation_loss(self.s, self.t, divergence="js", topk=4)
        self.assertTrue(torch.isfinite(val))
        self.assertGreaterEqual(float(val) + 1e-6, 0.0)


class TestBuildReftRow(unittest.TestCase):
    def test_l1_alignment(self):
        """Leading pad + (+1) shift + prompt masking, matching ReftDataset."""
        row = build_reft_row(
            input_ids=[10, 11, 12, 20, 21],
            prompt_len=3,
            prompt_ids=[10, 11, 12],
            word_id=5,
            pad_token_id=0,
            position="l1",
        )
        self.assertEqual(row["input_ids"].tolist(), [0, 10, 11, 12, 20, 21])
        self.assertEqual(
            row["labels"].tolist(),
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21],
        )
        self.assertEqual(row["attention_mask"].tolist(), [0, 1, 1, 1, 1, 1])
        # l1 selects the last prompt token (index 2), shifted +1 -> 3
        self.assertEqual(row["intervention_locations"], [[3]])
        self.assertEqual(row["subspaces"].tolist(), [[5]])

    def test_target_positions_recoverable_from_labels(self):
        row = build_reft_row(
            input_ids=[1, 2, 3, 4],
            prompt_len=2,
            prompt_ids=[1, 2],
            word_id=0,
            pad_token_id=0,
            position="l1",
        )
        labels = row["labels"]
        tgt = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        # y tokens (3,4) live at the tail; predictive positions are tgt-1
        self.assertEqual(row["input_ids"][tgt].tolist(), [3, 4])


class TestSDPOHelpers(unittest.TestCase):
    def test_trim_generated_stops_at_eos(self):
        self.assertEqual(_trim_generated([7, 8, 2, 9], eos_id=2, pad_id=0), [7, 8, 2])

    def test_trim_generated_stops_at_pad(self):
        self.assertEqual(_trim_generated([7, 0, 9], eos_id=2, pad_id=0), [7])

    def test_trim_generated_no_eos(self):
        self.assertEqual(_trim_generated([7, 8, 9], eos_id=2, pad_id=0), [7, 8, 9])

    def test_left_pad_shapes_and_mask(self):
        ids, attn = _left_pad([[1, 2], [3, 4, 5]], pad_id=0, device="cpu")
        self.assertEqual(ids.tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(attn.tolist(), [[0, 1, 1], [1, 1, 1]])

    def test_gather_next_token_logits(self):
        V = 4
        logits = torch.arange(5 * V, dtype=torch.float32).reshape(1, 5, V)
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 2, 3]])
        sel = _gather_next_token_logits(logits, labels)
        self.assertEqual(len(sel), 1)
        # labeled positions 3,4 -> predictive logits at 2,3
        self.assertTrue(torch.equal(sel[0], logits[0, [2, 3], :]))

    def test_gather_returns_none_without_labels(self):
        logits = torch.randn(1, 4, 3)
        labels = torch.full((1, 4), IGNORE_INDEX)
        self.assertIsNone(_gather_next_token_logits(logits, labels)[0])

    def test_gather_excludes_position_zero(self):
        logits = torch.randn(1, 3, 2)
        labels = torch.tensor([[5, IGNORE_INDEX, 6]])  # pos 0 labeled (no pos -1)
        sel = _gather_next_token_logits(logits, labels)
        # only position 2 is usable -> predictive logits at 1
        self.assertTrue(torch.equal(sel[0], logits[0, [1], :]))


class _RaisingModel:
    """Stand-in base model whose generate must never be called."""

    def generate(self, *a, **k):  # pragma: no cover - asserted not called
        raise AssertionError("generate() should not be called when words are cached")


class TestOffPolicyCache(unittest.TestCase):
    def setUp(self):
        self.cfg = SDPOConfig(n_offpolicy=2, offpolicy_pool=4)

    def test_draw_respects_n_and_pool_size(self):
        cache = {0: [[1], [2], [3]], 1: [[9]]}
        recs = _draw_offpolicy([0, 1, 0], cache, n=2, seed=0)
        per_word = {}
        for w, _ in recs:
            per_word[w] = per_word.get(w, 0) + 1
        # word 0 appears twice in the batch, 2 draws each; word 1 has only 1 in pool
        self.assertEqual(per_word, {0: 4, 1: 1})
        for w, y in recs:
            self.assertIn(y, cache[w])

    def test_draw_skips_empty_pool(self):
        cache = {5: []}
        self.assertEqual(_draw_offpolicy([5], cache, n=2, seed=0), [])

    def test_draw_is_seed_deterministic(self):
        cache = {0: [[1], [2], [3], [4], [5]]}
        a = _draw_offpolicy([0], cache, n=2, seed=123)
        b = _draw_offpolicy([0], cache, n=2, seed=123)
        self.assertEqual(a, b)

    def test_ensure_skips_generation_when_cached(self):
        cache = {0: [[1]], 1: [[2]]}
        # All batch words already cached -> generate() must not be invoked.
        _ensure_offpolicy_cached(
            _RaisingModel(),
            teacher_prompt_ids=[[0], [0]],
            word_ids=[0, 1, 0],
            cache=cache,
            cfg=self.cfg,
            tokenizer=None,
            device="cpu",
        )
        self.assertEqual(set(cache.keys()), {0, 1})


class TestDedupeRecords(unittest.TestCase):
    def test_drops_repeated_word_sequence_pairs(self):
        recs = [(0, [8, 9]), (0, [8, 9]), (1, [8, 9]), (0, [8])]
        self.assertEqual(
            _dedupe_records(recs), [(0, [8, 9]), (1, [8, 9]), (0, [8])]
        )

    def test_same_sequence_different_word_kept(self):
        self.assertEqual(
            _dedupe_records([(0, [1]), (1, [1])]), [(0, [1]), (1, [1])]
        )


class TestComputeSDPOLoss(unittest.TestCase):
    """Orchestration test for compute_sdpo_loss via the gold-only path (no generation)."""

    def _run(self, divergence="forward_kl"):
        V = 12
        iv = _FakeIntervenable(V)
        base = _FakeBase(V)
        cfg = SDPOConfig(
            divergence=divergence,
            n_onpolicy=0,
            n_offpolicy=0,
            include_gold=True,
            position="l1",
        )
        return iv, compute_sdpo_loss(
            intervenable=iv,
            base_model=base,
            tokenizer=_FakeTokenizer(),
            word_ids=[0, 1],
            student_prompt_ids=[2, 3],
            teacher_prompt_ids=[[4, 5, 6], [4, 7, 6]],
            target_ids=[[8, 9], [8]],
            cfg=cfg,
            collator=_fake_collator,
            device="cpu",
        )

    def test_metrics_and_finite_loss(self):
        _, (loss, metrics) = self._run()
        self.assertTrue(torch.isfinite(loss))
        # Two gold sequences: lengths 2 and 1 -> 2 rows, 3 distilled tokens.
        self.assertEqual(metrics["sdpo_n_seq"], 2.0)
        self.assertEqual(metrics["sdpo_n_tokens"], 3.0)

    def test_gradient_reaches_student_only(self):
        iv, (loss, _) = self._run()
        loss.backward()
        self.assertIsNotNone(iv.scale.grad)
        self.assertGreater(float(iv.scale.grad.abs()), 0.0)

    def test_all_divergences_run(self):
        for div in ("forward_kl", "reverse_kl", "js"):
            _, (loss, _) = self._run(div)
            self.assertTrue(torch.isfinite(loss), msg=div)

    def test_no_source_returns_zero(self):
        cfg = SDPOConfig(n_onpolicy=0, n_offpolicy=0, include_gold=False)
        loss, metrics = compute_sdpo_loss(
            intervenable=_FakeIntervenable(5),
            base_model=_FakeBase(5),
            tokenizer=_FakeTokenizer(),
            word_ids=[0],
            student_prompt_ids=[2],
            teacher_prompt_ids=[[3]],
            target_ids=[[4]],
            cfg=cfg,
            collator=_fake_collator,
            device="cpu",
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(metrics["sdpo_n_tokens"], 0.0)


class TestSharedBias(unittest.TestCase):
    """The per-step shared-b store makes CE and SDPO reuse the same sampled bias."""

    def _iv(self):
        return DistributionalWordIntervention(
            embed_dim=8,
            low_rank_dimension=4,
            num_words=3,
            dtype=torch.float32,
            device="cpu",
        )

    def test_shared_store_reuses_same_b_per_word(self):
        iv = self._iv()
        iv._shared_b = {}
        mu = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        wid = torch.tensor([0, 1])
        b1 = iv._bias_sample_rows(mu, logvar, wid)
        b2 = iv._bias_sample_rows(mu, logvar, wid)  # same words -> reuse cached b
        self.assertTrue(torch.equal(b1, b2))
        self.assertEqual(set(iv._shared_b.keys()), {0, 1})

    def test_no_store_samples_fresh(self):
        iv = self._iv()
        iv._shared_b = None
        mu = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        wid = torch.tensor([0, 1])
        b1 = iv._bias_sample_rows(mu, logvar, wid)
        b2 = iv._bias_sample_rows(mu, logvar, wid)
        self.assertFalse(torch.equal(b1, b2))

    def test_shared_b_retains_grad_path(self):
        """Cached b keeps its grad path, so reuse shares gradients back to mu."""
        iv = self._iv()
        iv._shared_b = {}
        mu = torch.zeros(2, 4, requires_grad=True)
        logvar = torch.zeros(2, 4)
        wid = torch.tensor([0, 1])
        # First call samples + caches b (with grad); a second call reuses the same
        # tensors, and backprop through the reuse reaches mu via the shared node.
        iv._bias_sample_rows(mu, logvar, wid)
        for b_row in iv._shared_b.values():
            self.assertTrue(b_row.requires_grad)
        reused = iv._bias_sample_rows(mu, logvar, wid)
        reused.sum().backward()
        self.assertIsNotNone(mu.grad)
        self.assertTrue(torch.allclose(mu.grad, torch.ones_like(mu.grad)))


class TestSharedBiasNetwork(unittest.TestCase):
    """Per-step ``_shared_bias`` avoids recomputing bias-network rows in SDPO."""

    def _iv_with_bias_network(self):
        embed_cache = torch.randn(4, 8)
        return DistributionalWordIntervention(
            embed_dim=16,
            low_rank_dimension=4,
            num_words=4,
            add_bias_network=True,
            embed_cache=embed_cache,
            dtype=torch.float32,
            device="cpu",
        )

    def test_shared_bias_reuses_mu_logvar(self):
        iv = self._iv_with_bias_network()
        iv._shared_bias = {}
        wid = torch.tensor([0, 1])
        mu1, lv1 = iv.get_bias_mu_logvar(wid)
        cached_mu = {w: iv._shared_bias[w]["mu"] for w in (0, 1)}
        cached_lv = {w: iv._shared_bias[w]["logvar"] for w in (0, 1)}
        mu2, lv2 = iv.get_bias_mu_logvar(wid)
        self.assertTrue(torch.equal(mu1, mu2))
        self.assertTrue(torch.equal(lv1, lv2))
        self.assertEqual(set(iv._shared_bias.keys()), {0, 1})
        for w in (0, 1):
            self.assertIs(iv._shared_bias[w]["mu"], cached_mu[w])
            self.assertIs(iv._shared_bias[w]["logvar"], cached_lv[w])

    def test_shared_bias_retains_grad_path(self):
        iv = self._iv_with_bias_network()
        iv._shared_bias = {}
        wid = torch.tensor([0, 1])
        iv.get_bias_mu_logvar(wid)
        reused_mu, _ = iv.get_bias_mu_logvar(wid)
        reused_mu.sum().backward()
        self.assertIsNotNone(iv.bias_network.fc1.weight.grad)
        self.assertGreater(float(iv.bias_network.fc1.weight.grad.abs().sum()), 0.0)

    def test_vae_bias_network_sdpo_caches_are_separate(self):
        """CE forward: bias-network cache then sampled-b cache (regression for shared dict)."""
        iv = self._iv_with_bias_network()
        shared = {}
        iv._shared_b = shared
        iv._shared_bias = shared  # old bug: same dict object
        wid = torch.tensor([0, 1])
        with self.assertRaises(TypeError):
            mu, logvar = iv.get_bias_mu_logvar(wid)
            iv._bias_sample_rows(mu, logvar, wid)

        iv._shared_b = {}
        iv._shared_bias = {}
        mu, logvar = iv.get_bias_mu_logvar(wid)
        b = iv._bias_sample_rows(mu, logvar, wid)
        self.assertEqual(b.shape, (2, 4))
        self.assertIsInstance(iv._shared_b[0], torch.Tensor)
        self.assertIsInstance(iv._shared_bias[0], dict)

    def test_llm_encoder_forward_once_per_word_per_step(self):
        from unittest.mock import MagicMock

        from boreft.pyreft.interventions import DistributionalWordIntervention

        penult = torch.randn(3, 5, 16)
        attn = torch.ones(3, 5)
        encoder = MagicMock(return_value=torch.randn(2, 8))
        iv = DistributionalWordIntervention(
            embed_dim=16,
            low_rank_dimension=4,
            num_words=3,
            add_bias_network=True,
            bias_input_source="llm_encoder",
            bias_network_input_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        iv.set_semantic_encoder(encoder)
        iv.set_encoder_inputs(penult, attn)
        iv._shared_bias = {}
        wid = torch.tensor([0, 1])
        iv.get_bias_mu_logvar(wid)
        iv.get_bias_mu_logvar(wid)
        self.assertEqual(encoder.call_count, 1)


class TestInterventionIndexAlignment(unittest.TestCase):
    """Generation and scoring must intervene on the same semantic token."""

    def _check(self, position):
        prompt_ids = [11, 12, 13]
        y = [20, 21]
        pos_gen = intervention_position_list(position, prompt_ids)
        row = build_reft_row(
            input_ids=prompt_ids + y,
            prompt_len=len(prompt_ids),
            prompt_ids=prompt_ids,
            word_id=0,
            pad_token_id=0,
            position=position,
        )
        pos_score = row["intervention_locations"][0]
        # Scoring inserts a leading pad, so indices shift by exactly +1 ...
        self.assertEqual(pos_score, [p + 1 for p in pos_gen])
        # ... and therefore point at the identical underlying prompt token.
        padded = row["input_ids"].tolist()
        for pg, ps in zip(pos_gen, pos_score):
            self.assertEqual(padded[ps], prompt_ids[pg])

    def test_l1(self):
        self._check("l1")

    def test_f1(self):
        self._check("f1")


# Toy real-model integration (runs on the cluster; skipped where deps are absent).
try:
    from transformers import GPT2Config, GPT2LMHeadModel

    from boreft.pyreft import (
        LoreftPerWordBiasIntervention,
        ReftConfig,
        build_reft_representation,
        get_reft_model,
    )

    _HAS_REFT_DEPS = True
except Exception:  # pragma: no cover - import guard
    _HAS_REFT_DEPS = False


@unittest.skipUnless(_HAS_REFT_DEPS, "transformers/pyvene not available")
class TestSDPOToyModelIntegration(unittest.TestCase):
    def test_generation_scoring_and_backprop(self):
        torch.manual_seed(0)
        V = 64
        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=V, n_positions=64, n_embd=32, n_layer=2, n_head=2)
        )
        model.eval()
        iv = LoreftPerWordBiasIntervention(
            embed_dim=32,
            low_rank_dimension=4,
            num_words=2,
            dtype=torch.float32,
            device="cpu",
        )
        reft = get_reft_model(
            model,
            ReftConfig(representations=[build_reft_representation(1, 4, iv)]),
        )
        cfg = SDPOConfig(
            divergence="forward_kl",
            sample_temperature=1.0,
            sample_top_p=1.0,
            max_new_tokens=3,
            n_onpolicy=2,
            n_offpolicy=0,
            include_gold=True,
            position="l1",
        )
        loss, metrics = compute_sdpo_loss(
            intervenable=reft,
            base_model=model,
            tokenizer=_FakeTokenizer(),
            word_ids=[0, 1],
            student_prompt_ids=[5, 6, 7],
            teacher_prompt_ids=[[5, 6, 7], [5, 6, 7]],
            target_ids=[[9, 10], [11]],
            cfg=cfg,
            collator=_fake_collator,
            device="cpu",
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["sdpo_n_tokens"], 0.0)

        loss.backward()
        iv_grads = [p.grad for p in iv.parameters() if p.grad is not None]
        self.assertTrue(iv_grads, "intervention params received no gradient")
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in iv_grads))

    def _ce_like_forward(self, reft, word_ids, prompt_ids, target_ids):
        """Run an intervened forward over [prompt; target] (mimics the CE pass)."""
        rows = [
            build_reft_row(
                input_ids=list(prompt_ids) + list(target_ids[w]),
                prompt_len=len(prompt_ids),
                prompt_ids=prompt_ids,
                word_id=w,
                pad_token_id=0,
                position="l1",
            )
            for w in word_ids
        ]
        batch = _fake_collator(rows)
        unit_locations = {
            "sources->base": (
                None,
                batch["intervention_locations"].permute(1, 0, 2).tolist(),
            )
        }
        reft(
            {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
            unit_locations=unit_locations,
            labels=batch["labels"],
            subspaces=batch["subspaces"].permute(1, 0, 2).tolist(),
        )

    def test_vae_shared_b_reused_in_sdpo(self):
        """Full VAE path: SDPO reuses the CE-sampled b and shares its grad path."""
        torch.manual_seed(0)
        V = 64
        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=V, n_positions=64, n_embd=32, n_layer=2, n_head=2)
        )
        iv = DistributionalWordIntervention(
            embed_dim=32,
            low_rank_dimension=4,
            num_words=2,
            dtype=torch.float32,
            device="cpu",
        )
        reft = get_reft_model(
            model,
            ReftConfig(representations=[build_reft_representation(1, 4, iv)]),
        )
        iv.train()

        word_ids = [0, 1]
        prompt_ids = [5, 6, 7]
        target_ids = [[9, 10], [11]]
        teacher_prompt_ids = [[5, 6, 7], [5, 6, 7]]

        # Trainer-like: open the shared-b store, then a CE-like forward populates it.
        iv._shared_b = {}
        iv._shared_bias = {}
        self._ce_like_forward(reft, word_ids, prompt_ids, target_ids)
        self.assertEqual(set(iv._shared_b), {0, 1})
        cached = dict(iv._shared_b)  # exact tensor objects sampled by CE

        cfg = SDPOConfig(
            divergence="forward_kl",
            n_onpolicy=0,
            n_offpolicy=0,
            include_gold=True,
            position="l1",
        )
        loss, metrics = compute_sdpo_loss(
            intervenable=reft,
            base_model=model,
            tokenizer=_FakeTokenizer(),
            word_ids=word_ids,
            student_prompt_ids=prompt_ids,
            teacher_prompt_ids=teacher_prompt_ids,
            target_ids=target_ids,
            cfg=cfg,
            collator=_fake_collator,
            device="cpu",
        )
        # SDPO scoring reused the cached b (no resample): same keys, same objects.
        self.assertEqual(set(iv._shared_b), {0, 1})
        for w in word_ids:
            self.assertIs(iv._shared_b[w], cached[w])
        self.assertTrue(torch.isfinite(loss))

        # The SDPO gradient reaches word_mu through the shared (CE-sampled) b.
        loss.backward()
        self.assertIsNotNone(iv.word_mu.weight.grad)
        self.assertGreater(float(iv.word_mu.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
