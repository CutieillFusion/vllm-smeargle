# SPDX-License-Identifier: Apache-2.0
"""SMEARGLE draft model for speculative decoding with a Llama target.

Architecture mirrors EAGLE-3 (single-layer draft, embeds + target hidden_state
concat at the input, separate small lm_head with d2t/t2d vocab translation),
but the decoder block is a **Mamba2 SSM** instead of self-attention. See
neurips2026-smeargle/smeargle:testsmeargle/model/cnets.py for the reference
implementation we mirror.

Source layout the converter produces (see scripts/convert-smeargle-for-vllm.py):
  config.json    architectures=["Mamba2LlamaSmeargleForCausalLM"], model_type="llama"
                 with smeargle extras: state_size, conv_kernel, n_groups, expand,
                 chunk_size, draft_vocab_size, target_hidden_size
  model.safetensors   embed_tokens, midlayer.{hidden_norm,input_layernorm,post_attention_layernorm}
                      midlayer.mamba2.{mamba2.*, out_proj.weight}
                      midlayer.mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
                      fc.weight, norm.weight, lm_head.weight, d2t, t2d

The reference uses HuggingFace's Mamba2Block; we use the vendored implementation
from transformers since vLLM's MambaMixer2 is tightly bound to vLLM's batched
inference cache manager (we want to manage cache state ourselves for tree-spec).
"""
from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.activations import ACT2FN

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.multimodal.inputs import NestedTensors

# Optional trace dumper for cross-impl comparison with standalone.
try:
    import sys as _sys
    _sys.path.insert(0, "/tmp/smtrace")
    from smeargle_trace import trace as _smtrace  # type: ignore
except Exception:
    def _smtrace(*_a, **_k):  # type: ignore
        return False

from .utils import AutoWeightsLoader, get_draft_quant_config, maybe_prefix

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Mamba2 sub-blocks
#
# We re-use HF transformers' Mamba2Block at runtime (it's already installed in
# the NVIDIA vLLM image, matches the reference exactly, and avoids needing to
# graft vLLM's batched MambaMixer2 onto tree-spec cache semantics).
# ---------------------------------------------------------------------------


def _build_hf_mamba2_block(draft_config: LlamaConfig, target_hidden: int,
                           num_attention_heads: int, rms_norm_eps: float):
    """Construct an HF Mamba2Block with the same hyperparameters the SMEARGLE
    reference uses (see cnets.py Mamba2.__init__).

    Returns (mamba2_block, hf_mamba2_config).
    """
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Block

    hf_cfg = Mamba2Config(
        hidden_size=target_hidden * 2,                # concat(embed, hidden) input
        num_heads=num_attention_heads * 2,
        head_dim=target_hidden // num_attention_heads,
        state_size=getattr(draft_config, "state_size", 16),
        expand=getattr(draft_config, "expand", 1),
        conv_kernel=getattr(draft_config, "conv_kernel", 4),
        n_groups=getattr(draft_config, "n_groups", 1),
        chunk_size=getattr(draft_config, "chunk_size", 256),
        num_hidden_layers=1,
        residual_in_fp32=True,
        layer_norm_epsilon=rms_norm_eps,
    )
    return Mamba2Block(hf_cfg, layer_idx=0), hf_cfg


class _LlamaMLPWithBias(nn.Module):
    """Llama-style gated MLP that, unlike vLLM's, supports bias on the
    projections (the SMEARGLE checkpoint has gate/up/down biases)."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        h = config.hidden_size
        i = config.intermediate_size
        bias = getattr(config, "mlp_bias", True)
        self.gate_proj = nn.Linear(h, i, bias=bias)
        self.up_proj = nn.Linear(h, i, bias=bias)
        self.down_proj = nn.Linear(i, h, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act if hasattr(config, "hidden_act") else "silu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SmeargleDecoderLayer(nn.Module):
    """Single SMEARGLE draft layer: norm(embed) + norm(hidden) -> cat -> Mamba2 ->
    +residual -> norm + MLP -> +residual."""

    def __init__(self, draft_config: LlamaConfig, target_config: LlamaConfig):
        super().__init__()
        h = target_config.hidden_size
        self.hidden_size = h

        self.mamba2_block, self._hf_mamba2_config = _build_hf_mamba2_block(
            draft_config,
            target_hidden=h,
            num_attention_heads=target_config.num_attention_heads,
            rms_norm_eps=target_config.rms_norm_eps,
        )
        self.out_proj = nn.Linear(self._hf_mamba2_config.hidden_size, h, bias=False)
        self.mlp = _LlamaMLPWithBias(target_config)
        self.input_layernorm = RMSNorm(h, eps=target_config.rms_norm_eps)
        self.hidden_norm = RMSNorm(h, eps=target_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(h, eps=target_config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: torch.Tensor | None = None,
    ):
        # vLLM packs all tokens (across the batch and across spec-decode tree
        # branches) into a 2D tensor of shape (num_tokens, hidden). HF's
        # Mamba2Block expects (batch, seq_len, hidden) where the cache has
        # batch_size matching the input's batch_size.
        #
        # For spec decode, every token row is its OWN branch with its own
        # Mamba2Cache slot, so we want batch=num_tokens, seq_len=1 — NOT
        # batch=1, seq_len=num_tokens (which would share state across rows).
        squeezed_out = False
        if hidden_states.dim() == 2:
            # (N, H) -> (N, 1, H): each row is a separate cache slot
            hidden_states = hidden_states.unsqueeze(1)
            input_emb = input_emb.unsqueeze(1)
            squeezed_out = True

        residual = hidden_states

        h_n = self.hidden_norm(hidden_states)
        e_n = self.input_layernorm(input_emb)
        x = torch.cat((e_n, h_n), dim=-1)

        # Mamba2Block: disable cuDNN. Empirically the cuDNN path produces a
        # 1.51x-larger Mamba2 SSM cache magnitude than the ATen path on
        # Blackwell (sm_121), which corrupts the SSM state and collapses
        # acceptance. At B=1 cuDNN often fails to find an engine and falls
        # back silently; at B>1 it succeeds with the wrong values.
        with torch.backends.cudnn.flags(enabled=False):
            x = self.mamba2_block(
                hidden_states=x,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=None,
            )
        x = x.to(self.out_proj.weight.dtype)
        x = self.out_proj(x)

        hidden_states = residual + x

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if squeezed_out:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level draft model
# ---------------------------------------------------------------------------


class SmeargleModel(nn.Module):
    """The draft body: embed_tokens, optional fc (aux hidden-state fusion),
    midlayer (Mamba2 + MLP), final norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config: LlamaConfig = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        # SMEARGLE always sees a single normalized hidden state from the target's
        # last layer (the reference fc only fires when target gives 3 stacked
        # aux states). We keep both code paths to be EAGLE3-compatible.
        eagle_config = getattr(self.config, "eagle_config", None) or {}
        self.use_aux_hidden_state = bool(eagle_config.get("use_aux_hidden_state", True))
        self.norm_before_fc = getattr(self.config, "norm_before_fc", False)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.midlayer = SmeargleDecoderLayer(self.config, self.config)

        if self.use_aux_hidden_state:
            target_hidden = getattr(self.config, "target_hidden_size", self.config.hidden_size)
            fc_input_size = target_hidden * 3
            self.input_norm = (
                RMSNorm(fc_input_size, eps=self.config.rms_norm_eps)
                if self.norm_before_fc else None
            )
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )

        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,  # accepted for API parity; unused (Mamba2 doesn't use RoPE)
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        cache_params=None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        if hidden_states.shape[-1] != input_embeds.shape[-1]:
            # Aux hidden-state path: target gave us a stack of (B, T, 3*H)
            if self.norm_before_fc and self.input_norm is not None:
                hidden_states = self.input_norm(hidden_states)
            hidden_states = self.fc(hidden_states)

        out = self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
        )
        out = self.norm(out)
        return out, out  # match (hidden_states, hidden_prenorm) tuple shape

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Mirror the reference state-dict layout. The converter has already
        stripped `_orig_mod.` and saved as safetensors; we just need to handle
        the `midlayer.` prefix and SMEARGLE's specific submodule names."""
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            # The reference uses `midlayer.mamba2.mamba2.*` for the inner HF block;
            # we store it at `midlayer.mamba2_block.*` for clarity.
            if name.startswith("midlayer.mamba2.mamba2."):
                name = name.replace("midlayer.mamba2.mamba2.", "midlayer.mamba2_block.")
            elif name.startswith("midlayer.mamba2.out_proj."):
                name = name.replace("midlayer.mamba2.out_proj.", "midlayer.out_proj.")
            # The reference's self_attn.{q,k,v,o}_proj is from its EAGLE-style
            # layer; SMEARGLE checkpoints don't have those — they have Mamba2
            # + MLP-with-bias only. Don't try to remap attention weights here.
            if name not in params_dict:
                # Skip unknown keys but log noisily so we can fix the converter
                logger.warning("[smeargle] skipping unknown weight key: %s", name)
                continue
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, w)
            loaded.add(name)
        return loaded


# ---------------------------------------------------------------------------
# Causal-LM wrapper (the registered architecture)
# ---------------------------------------------------------------------------


class Mamba2LlamaSmeargleForCausalLM(LlamaForCausalLM):
    """Public architecture name. Registered in models/registry.py as
    "Mamba2LlamaSmeargleForCausalLM". vLLM routes via the `architectures` field
    in config.json, not model_type, so set model_type="llama" in the config
    to keep HF's auto-config happy."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config: LlamaConfig = vllm_config.speculative_config.draft_model_config.hf_config

        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = self.config.vocab_size

        # Some downstream pieces of vLLM (model_loader, etc.) read this.
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num

        self.model = SmeargleModel(vllm_config=vllm_config, prefix="model")

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        # vLLM's eagle pathway uses draft_id_to_target_id (renamed from d2t).
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

    # API parity with Eagle3LlamaForCausalLM ---------------------------------

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        cache_params=None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            cache_params=cache_params,
            cache_position=cache_position,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Apply lm_head, then translate draft vocab back to target vocab
        (identical to Eagle3LlamaForCausalLM)."""
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None or self.config.draft_vocab_size == self.config.vocab_size:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full((logits.shape[0], self.config.vocab_size), float("-inf"))
        logits_new[:, targets] = logits
        return logits_new

    def compute_target_argmax(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """Fast greedy: lm_head → argmax in 32K draft vocab → translate via
        ``d2t`` to target vocab. Matches the standalone reference (cnets.py)
        and avoids `compute_logits`' inflate-to-128K + scatter cost. Output
        shape: ``(batch, 1)`` of target-vocab token ids.
        """
        draft_logits = self.logits_processor(self.lm_head, hidden_states)
        if draft_logits is None:
            return None
        draft_argmax = draft_logits.argmax(dim=-1, keepdim=True)
        if (
            self.draft_id_to_target_id is None
            or self.config.draft_vocab_size == self.config.vocab_size
        ):
            return draft_argmax
        return (
            draft_argmax
            + self.draft_id_to_target_id[draft_argmax.view(-1)].view(draft_argmax.shape)
        )

    def compute_target_topk(
        self, hidden_states: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fast top-k: log-softmax + topk in 32K draft vocab → translate via
        ``d2t``. Returns ``(top_logprobs, top_target_ids)``, both shape
        ``(batch, k)``. Used by the beam-search seed step in the proposer.
        """
        draft_logits = self.logits_processor(self.lm_head, hidden_states)
        if draft_logits is None:
            return None
        log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
        top = torch.topk(log_probs, k, dim=-1)
        if (
            self.draft_id_to_target_id is None
            or self.config.draft_vocab_size == self.config.vocab_size
        ):
            return top.values, top.indices
        target_ids = top.indices + self.draft_id_to_target_id[top.indices]
        return top.values, target_ids

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress 3 aux hidden states into 1 via fc (EAGLE3-style). Called
        by the proposer before forward when the target hands us a stack."""
        if not self.model.use_aux_hidden_state:
            return hidden_states
        if self.model.norm_before_fc and self.model.input_norm is not None:
            hidden_states = self.model.input_norm(hidden_states)
        return self.model.fc(hidden_states)

    # Weight loader ----------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights: dict[str, torch.Tensor] = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, w in weights:
            if "t2d" in name:
                continue  # vLLM doesn't use the target-to-draft mask; loader builds it
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = w

        skip_substrs: list[str] = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        if not self.model.norm_before_fc:
            skip_substrs.append("input_norm.")
        loader = AutoWeightsLoader(self, skip_prefixes=None, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
