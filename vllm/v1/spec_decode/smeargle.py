# SPDX-License-Identifier: Apache-2.0
"""Standalone SMEARGLE speculative-decoding proposer.

Drives ``vllm.model_executor.models.llama_smeargle.Mamba2LlamaSmeargleForCausalLM``
through the spec-decode interface without inheriting any of the attention
plumbing in ``EagleProposer``. The gpu_model_runner has a dedicated
``elif method == 'smeargle'`` branch that calls our ``propose()`` with the
minimum inputs we actually need:

  - the aux hidden-state stack from the target's three forwarded layers
  - the just-verified next-token id per request
  - per-request sequence offsets so we can isolate the last-token row

Why beam search emits a chain (rather than a tree):
  vLLM 0.19's rejection sampler is chain-only — it walks the draft list
  position-by-position and stops at the first reject (see
  ``rejection_greedy_sample_kernel``). So even though there's a
  ``TreeAttentionBackend`` that gives the target the correct logits for tree
  candidates, the sampler ignores the tree topology. Emitting a literal tree
  buys nothing.
  Instead we do tree exploration *internally* (beam search of width ``K`` to
  depth ``num_speculative_tokens``) and emit the highest-cumulative-log-prob
  chain. The verifier sees a chain; we still benefit from considering K
  alternatives at each step.

Mamba2 cache lifecycle:
  - Start fresh, batch=B (one slot per request).
  - After initial forward: expand to ``B*K`` (one slot per beam).
  - Each step: forward all beams, take top-K per beam, then top-K cumulative
    over the K*K candidates, ``reindex`` Mamba states by winning-parent.

Falls back to chain top-1 (no beam search) when ``smeargle_beam_width=1``.
"""
from __future__ import annotations

import ast
import json
import os
import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher  # noqa: F401


# Mamba2 cache adapter: stores tensors in transformers-v4 layout (single big
# [num_layers, batch, ...] tensors at .conv_states / .ssm_states) so SMEARGLE's
# existing batch-dim helpers (_expand_cache, _reindex_cache, _build_batched_*,
# etc.) work unchanged, while exposing the transformers-v5 layered surface
# (cache.layers[i].conv_states/recurrent_states, has_previous_state,
# update_conv_state, update_recurrent_state) that HF's v5 Mamba2Block forward
# calls into. Transformers v5 removed Mamba2Cache entirely, so we always use
# our own implementation.
class _LayerView:
    """Per-layer view into a Mamba2Cache. Mirrors the v5 layered-cache
    surface that HF's Mamba2Mixer.torch_forward reads from."""

    def __init__(self, cache: "Mamba2Cache", layer_idx: int):
        self._cache = cache
        self._layer_idx = layer_idx

    @property
    def conv_states(self):
        return self._cache.conv_states[self._layer_idx]

    @conv_states.setter
    def conv_states(self, value):
        self._cache.conv_states[self._layer_idx] = value

    @property
    def recurrent_states(self):
        # v5 renamed ssm_states -> recurrent_states; storage stays in
        # self._cache.ssm_states under the v4 name for SMEARGLE compatibility.
        return self._cache.ssm_states[self._layer_idx]

    @recurrent_states.setter
    def recurrent_states(self, value):
        self._cache.ssm_states[self._layer_idx] = value

    @property
    def has_previous_state(self):
        return self._cache._has_prev[self._layer_idx]

    @property
    def is_conv_states_initialized(self):
        return self._cache._initialized[self._layer_idx]

    @property
    def is_recurrent_states_initialized(self):
        return self._cache._initialized[self._layer_idx]

    @property
    def device(self):
        return self._cache.conv_states.device

    @property
    def dtype(self):
        return self._cache.conv_states.dtype


class Mamba2Cache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)
        num_layers = config.num_hidden_layers

        self.conv_states = torch.zeros(
            num_layers,
            batch_size,
            self.intermediate_size + 2 * self.n_groups * self.state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_states = torch.zeros(
            num_layers,
            batch_size,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=device,
            dtype=dtype,
        )

        # Per-layer flags driving v5's prefill-vs-decode branch. Flipped
        # True the first time update_conv_state / update_recurrent_state
        # writes that layer.
        self._has_prev = [False] * num_layers
        self._initialized = [False] * num_layers

        # v5 surface: per-layer view objects.
        self.layers = [_LayerView(self, i) for i in range(num_layers)]

    # ---- v5 API surface (called by HF's Mamba2Mixer.torch_forward) ------
    def has_previous_state(self, layer_idx=None):
        if layer_idx is None:
            return any(self._has_prev)
        return self._has_prev[layer_idx]

    def update_conv_state(self, conv_states, layer_idx, **_kwargs):
        """v5-style update. Mirrors LinearAttentionLayer.update_conv_state's
        copy-on-first / roll-and-tail-on-subsequent behavior."""
        if not self._has_prev[layer_idx]:
            self.conv_states[layer_idx].copy_(conv_states)
        else:
            num_new = conv_states.shape[-1]
            if num_new >= self.conv_kernel_size:
                self.conv_states[layer_idx].copy_(
                    conv_states[..., -self.conv_kernel_size:]
                )
            else:
                self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(
                    shifts=-num_new, dims=-1
                )
                self.conv_states[layer_idx][..., -num_new:] = conv_states
        self._has_prev[layer_idx] = True
        self._initialized[layer_idx] = True
        return self.conv_states[layer_idx]

    def update_recurrent_state(self, recurrent_states, layer_idx, **_kwargs):
        """v5-style update. Stores into the slice referenced as
        recurrent_states by HF v5; same storage as v4's ssm_states."""
        self.ssm_states[layer_idx].copy_(recurrent_states)
        self._has_prev[layer_idx] = True
        self._initialized[layer_idx] = True
        return self.ssm_states[layer_idx]

    # ---- Utility shared with v4 ----------------------------------------
    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()
        for i in range(len(self._has_prev)):
            self._has_prev[i] = False
            self._initialized[i] = False

logger = init_logger(__name__)


# ----------------------------------------------------------------------
# Profiling: per-section + per-Mamba2-call breakdown of SmeargleProposer.propose().
# Enable with SMEARGLE_PROFILE=1. One JSON line per propose() call is
# appended to SMEARGLE_PROFILE_JSONL (default /tmp/smtrace/smeargle_profile.jsonl).
# Use /tmp/smtrace/analyze_smeargle_profile.py to aggregate.
# ----------------------------------------------------------------------
class _SmtProfiler:
    def __init__(self) -> None:
        self.enabled = int(os.environ.get("SMEARGLE_PROFILE", "0")) == 1
        self.path = os.environ.get(
            "SMEARGLE_PROFILE_JSONL", "/tmp/smtrace/smeargle_profile.jsonl"
        )
        self._fp = None
        self._cur: dict = {}
        self._t0: dict = {}
        self._mamba_log: list = []  # list of (seq_len, ms) for each forward this call
        if self.enabled:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                # Truncate at process start so each run produces a clean log.
                self._fp = open(self.path, "w")
            except Exception as exc:
                logger.warning("[smeargle_profile] couldn't open %s: %s", self.path, exc)
                self.enabled = False

    def reset(self) -> None:
        if not self.enabled:
            return
        self._cur = {}
        self._t0 = {}
        self._mamba_log = []

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self._t0[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if not self.enabled:
            return
        t0 = self._t0.pop(name, None)
        if t0 is None:
            return
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        # Accumulate (so re-entering a section sums correctly).
        self._cur[name + "_ms"] = self._cur.get(name + "_ms", 0.0) + ms

    def add_mamba_call(self, seq_len: int, ms: float) -> None:
        if not self.enabled:
            return
        self._mamba_log.append((int(seq_len), float(ms)))

    def flush(self) -> None:
        if not self.enabled or self._fp is None:
            return
        if self._cur or self._mamba_log:
            rec = dict(self._cur)
            if self._mamba_log:
                rec["mamba_calls"] = [
                    {"seq_len": sl, "ms": round(m, 3)} for sl, m in self._mamba_log
                ]
                rec["mamba_call_count"] = len(self._mamba_log)
                rec["mamba_total_ms"] = round(sum(m for _, m in self._mamba_log), 3)
            try:
                self._fp.write(json.dumps(rec) + "\n")
                self._fp.flush()
            except Exception:
                pass
        self.reset()


def _try_enable_mamba2_fast_path() -> bool:
    """Inject prebuilt CUDA kernels into HF v5's Mamba2 module globals.

    HF v5 normally fetches `causal_conv1d` + `mamba_ssm` via its own
    ``lazy_load_kernel`` hook, which requires the ``kernels`` Python package
    — that package's API is incompatible with the transformers 5.12 in this
    venv (the ``LayerRepository`` initialiser raises without ``version``).

    The prebuilt kernels are already on disk from earlier ``kernels`` use
    (HF Hub cache). We load them directly via importlib and overwrite the
    HF module globals (`causal_conv1d_fn`, `selective_state_update`, ...)
    plus flip ``is_fast_path_available = True``. ``cuda_kernels_forward``
    reads these globals at call time, so the next forward will take the
    fast path even though Mamba2Mixer.__init__ already set them to None.

    Must run *after* the draft model has been constructed (after all
    ``Mamba2Mixer.__init__`` calls), otherwise their inits will re-clobber
    the globals. On by default; set ``SMEARGLE_FAST_PATH=0`` to disable
    (e.g., to diff against the slow path).
    """
    if int(os.environ.get("SMEARGLE_FAST_PATH", "1")) != 1:
        return False

    import glob as _glob
    import importlib.util as _ilu
    import sys as _sys

    def _load(pattern: str, name: str):
        roots = _glob.glob(pattern)
        if not roots:
            return None
        root = roots[0]
        init = root + "/__init__.py"
        try:
            spec = _ilu.spec_from_file_location(
                name, init, submodule_search_locations=[root]
            )
            mod = _ilu.module_from_spec(spec)
            _sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as exc:
            logger.warning("[smeargle] fast-path: failed to load %s: %s", name, exc)
            return None

    home = os.path.expanduser("~")
    cc = _load(
        f"{home}/.cache/huggingface/hub/kernels--kernels-community--causal-conv1d"
        "/snapshots/*/build/torch211-cxx11-cu130-aarch64-linux",
        "_smeargle_causal_conv1d",
    )
    mssm = _load(
        f"{home}/.cache/huggingface/hub/kernels--kernels-community--mamba-ssm"
        "/snapshots/*/build/torch211-cxx11-cu130-aarch64-linux",
        "_smeargle_mamba_ssm",
    )
    if cc is None or mssm is None:
        logger.warning(
            "[smeargle] fast-path: prebuilt kernels not found on disk; "
            "staying on torch_forward",
        )
        return False

    required = {
        "causal_conv1d_fn": getattr(cc, "causal_conv1d_fn", None),
        "causal_conv1d_update": getattr(cc, "causal_conv1d_update", None),
        "selective_state_update": getattr(mssm, "selective_state_update", None),
        "mamba_chunk_scan_combined": getattr(mssm, "mamba_chunk_scan_combined", None),
        "mamba_split_conv1d_scan_combined": getattr(
            mssm, "mamba_split_conv1d_scan_combined", None
        ),
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        logger.warning("[smeargle] fast-path: missing functions: %s", missing)
        return False

    try:
        from transformers.models.mamba2 import modeling_mamba2 as _m2
        for k, v in required.items():
            setattr(_m2, k, v)
        _m2.is_fast_path_available = True
        logger.info(
            "[smeargle] Mamba2 fast path enabled "
            "(causal_conv1d + mamba_ssm injected)"
        )
    except Exception as exc:
        logger.warning("[smeargle] fast-path injection failed: %s", exc)
        return False

    # Wrap Mamba2Mixer.forward with two responsibilities:
    #   (a) per-call (seq_len, ms) profiling into the active proposer's profiler
    #       (no-op unless SMEARGLE_PROFILE=1)
    #   (b) "prefill from prior cache state" dispatch — when seq_len>1 AND
    #       has_previous_state(layer_idx) is True, the stock HF v5 Mamba2 has
    #       no path (decode is 1-token, prefill always seeds from zero). We
    #       drive mamba_chunk_scan_combined + causal_conv1d_fn directly with
    #       initial_states from the cache to support O(K) incremental catchup.
    try:
        from transformers.models.mamba2 import modeling_mamba2 as _m2b
        if not getattr(_m2b.Mamba2Mixer.forward, "_smtprof_wrapped", False):
            _orig = _m2b.Mamba2Mixer.forward

            def _wrapped_forward(self, hidden_states, *args, **kwargs):
                cache_params = kwargs.get("cache_params")
                if cache_params is None and len(args) >= 1:
                    cache_params = args[0]
                seq_len = (
                    hidden_states.shape[1] if hidden_states.dim() >= 2 else 1
                )
                # When seq_len > 1 AND the cache already holds prior state for
                # this layer, redirect to our prefill-from-state path; HF's
                # Mamba2Mixer always seeds the SSD scan from zero, which is
                # wrong for our incremental catchup.
                use_prefill_from_state = (
                    cache_params is not None
                    and seq_len > 1
                    and getattr(cache_params, "has_previous_state", None)
                    is not None
                    and cache_params.has_previous_state(self.layer_idx)
                )

                prof = _ACTIVE_PROFILER[0] if _ACTIVE_PROFILER else None
                profile_on = prof is not None and prof.enabled
                if profile_on:
                    torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                try:
                    if use_prefill_from_state:
                        return _mamba2_prefill_from_prior_state(
                            self, hidden_states, cache_params,
                        )
                    return _orig(self, hidden_states, *args, **kwargs)
                finally:
                    if profile_on:
                        torch.cuda.synchronize()
                        prof.add_mamba_call(
                            seq_len,
                            (time.perf_counter() - _t0) * 1000.0,
                        )

            _wrapped_forward._smtprof_wrapped = True
            _m2b.Mamba2Mixer.forward = _wrapped_forward
            logger.info(
                "[smeargle] Mamba2Mixer.forward wrapper installed "
                "(profiling + prefill-from-state)"
            )
    except Exception as exc:
        logger.warning("[smeargle] mixer wrapper install failed: %s", exc)

    return True


def _mamba2_prefill_from_prior_state(mixer, hidden_states, cache_params):
    """Run Mamba2Mixer prefill on `hidden_states` (batch, seq_len, hidden)
    starting from `cache_params.layers[layer_idx].(conv_states, recurrent_states)`
    instead of zero state. Mirrors `cuda_kernels_forward`'s prefill else-branch
    (transformers/models/mamba2/modeling_mamba2.py lines ~334-396) but seeds
    both the conv kernel and the SSD scan with the prior cache.

    Used by SMEARGLE's incremental catchup to advance the draft Mamba2 cache
    by exactly the newly-committed tokens (~1-2 per step) instead of re-running
    a full prefill over the cumulative history each step.
    """
    layer_idx = mixer.layer_idx
    batch_size, seq_len, _ = hidden_states.shape
    groups_time_state_size = mixer.n_groups * mixer.ssm_state_size

    # 1. Gated MLP projection
    projected_states = mixer.in_proj(hidden_states)
    d_mlp = (
        projected_states.shape[-1]
        - 2 * mixer.intermediate_size
        - 2 * mixer.n_groups * mixer.ssm_state_size
        - mixer.num_heads
    ) // 2
    _, _, gate, hidden_states_B_C, dt = projected_states.split(
        [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads],
        dim=-1,
    )

    layer = cache_params.layers[layer_idx]

    # 2. Causal conv1d with initial state.
    # The kernel wants:
    #   x:               (batch, channels, seq_len)
    #   initial_states:  (batch, channels, width-1)  — only the W-1 most recent
    #                    pre-conv input tokens (NOT the full W in our cache slot).
    # We avoid `return_final_states=True` (which adds a channel-last stride
    # requirement) and update the cache ourselves by concatenating prior input
    # tokens with the new ones and taking the last W: that's the true semantic
    # of v5 LinearAttentionLayer.conv_states ("the last W pre-conv inputs").
    from transformers.models.mamba2 import modeling_mamba2 as _m2
    causal_conv1d_fn = _m2.causal_conv1d_fn

    W = mixer.conv_kernel_size
    # x must be channel-last: shape (B, C, K) with stride(1)==1.
    # `hidden_states_B_C` from split is (B, K, C) contiguous, so a plain
    # transpose(1, 2) yields the channel-last view directly. NO `.contiguous()`
    # — that would re-pack as channel-first.
    hidden_states_B_C_cf = hidden_states_B_C.transpose(1, 2)  # channel-last (B, C, K)
    # initial_states also needs channel-last layout. Our cache stores it as
    # plain contiguous (B, C, W), so we round-trip through (B, W, C) to land
    # at stride(1)==1.
    prior_conv = layer.conv_states  # (B, C, W) channel-first
    prior_conv_cl = prior_conv.transpose(1, 2).contiguous().transpose(1, 2)
    initial_for_kernel = prior_conv_cl[..., -(W - 1):]  # (B, C, W-1) channel-last
    conv_out_t = causal_conv1d_fn(
        x=hidden_states_B_C_cf,
        weight=mixer.conv1d.weight.squeeze(1),
        bias=mixer.conv1d.bias,
        activation=mixer.activation,
        initial_states=initial_for_kernel,
    )  # (B, C, K)
    # Update cache: new state = last W pre-conv inputs from (prior + new).
    combined = torch.cat([prior_conv, hidden_states_B_C_cf], dim=-1)
    layer.conv_states = combined[..., -W:].contiguous()
    hidden_states_B_C = conv_out_t.transpose(1, 2)

    # 3. Split + SSM
    hidden_states_split, B_proj, C_proj = torch.split(
        hidden_states_B_C,
        [mixer.intermediate_size, groups_time_state_size, groups_time_state_size],
        dim=-1,
    )

    A = -torch.exp(mixer.A_log.float())
    dt_limit_kwargs = (
        {} if mixer.time_step_limit == (0.0, float("inf"))
        else {"dt_limit": mixer.time_step_limit}
    )

    mamba_chunk_scan_combined = _m2.mamba_chunk_scan_combined
    prior_rec = layer.recurrent_states.contiguous()
    scan_output, new_rec = mamba_chunk_scan_combined(
        hidden_states_split.view(batch_size, seq_len, -1, mixer.head_dim),
        dt,
        A,
        B_proj.view(batch_size, seq_len, mixer.n_groups, -1),
        C_proj.view(batch_size, seq_len, mixer.n_groups, -1),
        chunk_size=mixer.chunk_size,
        D=mixer.D,
        z=None,
        seq_idx=None,
        return_final_states=True,
        initial_states=prior_rec,
        dt_bias=mixer.dt_bias,
        dt_softplus=True,
        **dt_limit_kwargs,
    )
    layer.recurrent_states = new_rec  # update cache

    scan_output = scan_output.view(batch_size, seq_len, -1)
    scan_output = mixer.norm(scan_output, gate)
    return mixer.out_proj(scan_output)


# Module-level slot so the wrapped Mamba2Mixer.forward can find the current
# proposer's profiler. Set by propose() at entry, cleared on exit.
_ACTIVE_PROFILER: list = []




class SmeargleProposer:
    """Owns the SMEARGLE draft model and runs chain speculation."""

    # ------------------------------------------------------------------
    # Construction & stub surface the gpu_model_runner reads off `drafter`
    # ------------------------------------------------------------------

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.runner = runner

        # Surface fields the runner reads. We declare them so any attribute
        # access (e.g. `self.drafter.kv_cache_gid`) just works without us
        # having to inherit EagleProposer's huge init.
        self.eagle3_use_aux_hidden_state: bool = True
        self.supports_mm_inputs: bool = False
        self.kv_cache_gid: int = -1
        self.parallel_drafting: bool = False
        self.draft_attn_groups: list = []  # intentionally empty
        self.hidden_size: int = self.draft_model_config.get_hidden_size()

        # Filled in by load_model().
        self.model: nn.Module | None = None
        # Cached HF Mamba2Config from the draft model (used to build per-call caches)
        self._mamba2_config = None

        # Beam width K for the internal tree exploration. K=1 collapses to
        # plain top-1 chain (current behaviour). K=2..4 explores a tree.
        # Read from env so we can tune without recompiling.
        import os
        # Chain decoding by default (beam_width=1). Set SMEARGLE_BEAM_WIDTH>1
        # to switch to beam search over K beams; only the chain path is
        # validated for production throughput.
        self.beam_width: int = int(os.environ.get("SMEARGLE_BEAM_WIDTH", "1"))

        # Chain-mode only: this proposer emits a chain of
        # ``num_speculative_tokens`` drafts. Tree mode was removed during the
        # cleanup that prepared this proposer for upstream; the chain path
        # was the only one validated at production scale.

        # Per-request stable Mamba2 cache. Mirrors the SMEARGLE reference's
        # `stable_cache_params` — instead of rebuilding a zero cache every
        # propose() call (which throws away all accumulated SSM/conv state
        # and tanks the draft model's accuracy), we keep one batch_size=1
        # Mamba2Cache per request_id and advance it through every newly
        # committed token (the accepted drafts of the previous step PLUS the
        # bonus/recovery, surfaced as next_token_ids).
        self.stable_caches: dict[str, "Mamba2Cache"] = {}
        # Set by gpu_model_runner before each propose() call so we can index
        # caches by request_id (batch composition changes step-to-step).
        self._current_req_ids: list[str] | None = None
        # Stashed by gpu_model_runner from the just-finished sampler: per-
        # request which path won + how many drafts were accepted. Together
        # with this step's tree_retrieve_indices these tell us which of the
        # previous step's drafts were committed, so we can advance each
        # request's stable cache through their hidden states.
        # Stashed previous-step target hidden states (fused) and qsl.
        # Standalone has the target's hidden at every committed sequence
        # position via its full hidden_states tensor; vLLM only exposes the
        # current step's input fused. Stashing the previous step's fused
        # lets catchup use the TARGET's hidden (not draft propagating) at
        # each accepted draft's actual position. Bounded memory: O(B*T*H).
        self._prev_fused: torch.Tensor | None = None
        self._prev_qsl: torch.Tensor | None = None
        # Per-request slot count from the previous step's qsl. The catchup
        # extension uses this to detect whether THIS request was prefilled
        # last step (slots > K+1) vs decoded (slots == K+1). The flat
        # `_prev_qsl[-1] > K+1` heuristic is wrong for B>1 because a B=2
        # decode has 2*(K+1) > K+1 — it would always misread as prefill and
        # the extension would never run.
        self._prev_qsl_per_rid: dict[str, int] = {}
        # Hidden state that "predicts NEXT after bonus", produced either by
        # the catchup's last forward (regular decode steps) or by
        # _maybe_prefill_new_requests' shifted prefill (first decode step).
        # The proposer iter loop uses this as iter-0's starting cur_hidden,
        # eliminating the redundant verified_forward that would otherwise
        # double-process the bonus.
        self._iter0_cur_hidden: torch.Tensor | None = None
        # FULL_REFRESH: per-request cumulative input_ids + target hiddens.
        # At every propose, rebuild stable cache via SSD prefill over the
        # entire cumulative sequence — mirrors standalone topK_genrate's
        # `stable_cache_params is None` else-branch behavior, applied at
        # every step rather than just the first. Memory: O(L*H) per request.
        self._cumul_input_ids: dict[str, list[int]] = {}
        self._cumul_hiddens: dict[str, torch.Tensor] = {}
        # Tracks cumul_input_ids length AFTER the last successful catchup. Used
        # by SMEARGLE_INCREMENTAL_CATCHUP=1 to compute the per-request "K new
        # tokens since last catchup" and feed only those into model.forward
        # starting from the prior cache state.
        self._prev_cumul_lens: dict[str, int] = {}
        # Saved Mamba2 (conv_states, recurrent_states) per request, captured
        # during the K=1 chain draft loop. Index 0 = state at loop entry
        # (= state through previous step's bonus). Index i = state after i
        # drafts have been advanced through the draft Mamba2. NEXT step's
        # catchup, given accept_len L from rejection sampling, picks
        # snapshots[L] as the new stable state and only needs to advance by
        # 1 token (this step's bonus). Replaces a multi-token catchup with a
        # single-token decode forward — gated by SMEARGLE_USE_DRAFT_SNAPSHOTS.
        #
        # The values are tuples of (conv_view, ssm_view) — VIEWS into the
        # per-request pre-allocated buffers in self._snapshot_buf. Allocating
        # buffers once per request (instead of cloning per iter) avoids dict
        # / tuple / tensor allocator churn each propose call.
        self._draft_snapshots: dict[str, list] = {}
        # Per-request pre-allocated snapshot buffers, keyed by req_id. Each
        # entry: {"conv": (depth+1, num_layers, 1, channels, kernel),
        #         "ssm":  (depth+1, num_layers, 1, num_heads, head_dim, state)}.
        # Lazy-allocated on first sight of each request; freed in
        # evict_stable_caches.
        self._snapshot_buf: dict[str, dict] = {}
        # Runner stashes THIS step's sampler output here just before propose
        # runs (= before _bookkeeping_sync updates num_tokens_no_spec).
        # FULL_REFRESH uses this to derive THIS step's accept_lens.
        self._latest_sampled_token_ids: torch.Tensor | None = None

        # Detailed timing profiler (SMEARGLE_PROFILE=1).
        self._smtprof = _SmtProfiler()

        logger.info(
            "[smeargle] proposer up (num_spec_tokens=%d, hidden=%d, "
            "beam_width=%d)",
            self.num_speculative_tokens, self.hidden_size, self.beam_width,
        )

    # All of these are called by the runner during init/validation. Mamba2
    # has no KV cache + no attention layers, so each is a no-op.
    def validate_same_kv_cache_group(self, kv_cache_config: "KVCacheConfig") -> None:
        return

    def initialize_attn_backend(self, kv_cache_config: "KVCacheConfig",
                                kernel_block_sizes=None) -> None:
        return

    def initialize_cudagraph_keys(self, cudagraph_mode) -> None:
        return

    # ------------------------------------------------------------------
    # Model load — mirror EAGLE3's two-step (build via get_model, wire up
    # target embeddings if shared) but without the attention-backend pieces.
    # ------------------------------------------------------------------

    def load_model(self, target_model: nn.Module) -> None:
        # Note on numerical determinism: without process-wide deterministic
        # algorithm pinning, cuBLAS/cuDNN may pick different GEMM/conv
        # algorithms for different (M, N, K) shapes — same input row, B=1
        # vs B=N → measurably different bf16 outputs. That noise feeds
        # through SMEARGLE's cumul_hiddens and reduces accept_len on
        # mixed-prompt batches by 5-10%. Operators who care can enable
        # process-wide determinism BEFORE launch with, e.g.:
        #     CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 \
        #     PYTHONHASHSEED=0 vllm serve ...
        # plus ``torch.use_deterministic_algorithms(True, warn_only=True)``
        # in a startup hook. SMEARGLE does not flip these globally itself
        # because the side effects affect the target model (and any other
        # model sharing the process) too.
        draft_vllm_config = self._draft_vllm_config()
        with set_default_torch_dtype(draft_vllm_config.model_config.dtype):
            self.model = get_model(
                vllm_config=draft_vllm_config, model_config=draft_vllm_config.model_config,
            )

        # If the converted draft head doesn't carry its own embed_tokens, share
        # the target's. The current SMEARGLE checkpoints DO carry embeddings
        # so this is a no-op in practice but keeps parity with the eagle path.
        with torch.no_grad():
            tgt_embed = _find_embed_tokens(target_model)
            if tgt_embed is not None and hasattr(self.model.model, "embed_tokens"):
                if self.model.model.embed_tokens.weight.shape == tgt_embed.weight.shape:
                    if not torch.equal(
                        self.model.model.embed_tokens.weight, tgt_embed.weight
                    ):
                        # Detected EAGLE-style independent embeddings; keep them.
                        pass

        # Cache the HF Mamba2Config carried on our decoder layer for fast cache
        # construction in propose().
        mid = self.model.model.midlayer
        self._mamba2_config = mid._hf_mamba2_config

        logger.info("[smeargle] draft model loaded; Mamba2 state_size=%d, expand=%d",
                    self._mamba2_config.state_size, self._mamba2_config.expand)

        _try_enable_mamba2_fast_path()

    # ------------------------------------------------------------------
    # Dummy run — touched at startup to size/shape-check things. We just
    # run one no-op forward against a 1-token cache to confirm everything
    # plugs in.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int, **kwargs) -> None:
        if self.model is None or self._mamba2_config is None:
            return
        h = self.hidden_size
        device = self.device
        dtype = self.dtype
        # Smallest possible run: 1 sequence, 1 token.
        ids = torch.zeros(1, dtype=torch.long, device=device)
        hs = torch.zeros(1, h, dtype=dtype, device=device)
        cache = self._fresh_cache(batch_size=1, device=device, dtype=dtype)
        cache_position = torch.tensor([0], device=device, dtype=torch.long)
        try:
            _ = self.model.forward(
                input_ids=ids,
                positions=torch.zeros(1, dtype=torch.long, device=device),
                hidden_states=hs,
                cache_params=cache,
                cache_position=cache_position,
            )
        except Exception as e:
            logger.warning("[smeargle] dummy_run hit %s: %s", type(e).__name__, e)

    # ------------------------------------------------------------------
    # The proposer's actual job: emit (num_reqs, total_tree_tokens)
    # draft token ids per request, following ``speculative_token_tree``.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propose(
        self,
        target_hidden_states: torch.Tensor,  # (num_tokens, 3*H) aux-concat or (num_tokens, H)
        next_token_ids: torch.Tensor,        # (num_reqs,) just-verified token ids
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        # Detailed profiler (gated by SMEARGLE_PROFILE=1 inside _SmtProfiler).
        self._smtprof.reset()
        # Publish for the Mamba2Mixer.forward wrapper to find.
        if _ACTIVE_PROFILER:
            _ACTIVE_PROFILER[0] = self._smtprof
        else:
            _ACTIVE_PROFILER.append(self._smtprof)
        self._smtprof.start("propose_total")
        self._smtprof.start("setup")

        """Chain-mode draft generation → emit ``num_speculative_tokens``-long chain per request.

        Returns ``(B, num_speculative_tokens)`` long tensor. vLLM's chain-only
        rejection sampler then walks it position-by-position.
        """
        assert self.model is not None
        device = next_token_ids.device

        depth = self.num_speculative_tokens
        K = max(1, self.beam_width)

        # --- Build the per-request last-hidden vector ------------------
        if target_hidden_states.shape[-1] != self.hidden_size:
            fused = self.model.combine_hidden_states(target_hidden_states)
        else:
            fused = target_hidden_states  # already fused

        qsl = common_attn_metadata.query_start_loc
        last_idx = (qsl[1:] - 1).to(torch.long)
        last_hidden = fused.index_select(0, last_idx)                  # (B, H)

        B = last_hidden.shape[0]

        # Per-request prompt prefill, cumulative state tracking, SSD-prefill
        # catchup at every step, plus the iter-loop wiring (skip the
        # redundant verified_forward, feed the catchup's last output as
        # iter-0 hidden) are the four pieces that together match the
        # standalone reference's ``mean_accept_len``. Verified by per-env
        # bisect; each piece is required.
        prefilled_last_hidden: dict[int, torch.Tensor] = (
            self._maybe_prefill_new_requests(
                fused=fused, qsl=qsl, device=device,
                next_token_ids=next_token_ids,
            )
        )
        # Initialize cumulative input_ids/hiddens for new requests so
        # _full_refresh_catch_up can extend them at subsequent steps.
        if self._current_req_ids and self.runner is not None:
            try:
                input_ids_gpu = self.runner.input_ids.gpu
                qsl_cpu = qsl.cpu().tolist()
                for b, req_id in enumerate(self._current_req_ids):
                    if req_id in self._cumul_input_ids:
                        continue  # already initialized; subsequent steps extend
                    start = qsl_cpu[b]
                    end = qsl_cpu[b + 1]
                    if end <= start or next_token_ids is None:
                        continue
                    prompt_ids = input_ids_gpu[start:end].long().tolist()
                    bonus_token = int(next_token_ids[b].item())
                    self._cumul_input_ids[req_id] = prompt_ids + [bonus_token]
                    # cumul_hiddens: target hiddens at prompt positions +
                    # one more for the bonus (= duplicate of last prompt
                    # hidden, since that's what predicted the bonus).
                    prompt_h = fused[start:end].detach().clone()
                    bonus_h = fused[end - 1:end].detach().clone()
                    self._cumul_hiddens[req_id] = torch.cat(
                        [prompt_h, bonus_h], dim=0,
                    )
            except Exception:
                pass

        # ---- Stable cache catch-up: rebuild per-request Mamba2 cache via
        # SSD prefill of the entire cumulative committed sequence each step.
        # Mirrors the standalone reference's catchup loop. Requires
        # _prev_fused/_prev_qsl stashed at end of the previous propose().
        out_hidden_from_catchup: torch.Tensor | None = None
        all_just_prefilled = (
            len(prefilled_last_hidden) == B and B > 0
        )
        if (
            self._current_req_ids is not None
            and len(self._current_req_ids) == B
            and not all_just_prefilled
            and self._prev_fused is not None
            and self._prev_qsl is not None
            and self._prev_qsl.shape[0] == B + 1
        ):
            self._smtprof.end("setup")
            self._smtprof.start("catchup")
            out_hidden_from_catchup = self._full_refresh_catch_up(
                fused=fused, qsl=qsl, B=B, device=device,
                next_token_ids=next_token_ids,
            )
            self._smtprof.end("catchup")
            self._smtprof.start("setup")  # remaining post-catchup setup
        catchup_ran = out_hidden_from_catchup is not None
        # When the catchup ran, the iter loop must skip its verified_forward
        # (the bonus is already in cache) and use the catchup's last output
        # as iter-0 hidden. Both wirings are paired — neither helps alone.
        self._skip_verified_forward = catchup_ran
        self._iter0_cur_hidden = out_hidden_from_catchup if catchup_ran else None

        # Stash this step's fused/qsl so the NEXT propose's catchup can
        # detect prefill (via _prev_qsl) and read target hiddens at PREV
        # input positions.
        self._prev_fused = fused.detach().clone()
        self._prev_qsl = qsl.detach().clone()
        if self._current_req_ids is not None:
            qsl_cpu = qsl.cpu().tolist()
            self._prev_qsl_per_rid = {
                rid: int(qsl_cpu[b + 1] - qsl_cpu[b])
                for b, rid in enumerate(self._current_req_ids)
                if b + 1 < len(qsl_cpu)
            }

        # Standalone (cnets.py:403) uses ``self.norm(out_hidden[:, -1])`` —
        # the DRAFT's last output after running the prompt + accepted tokens
        # through. vLLM's default ``last_hidden = fused.index_select`` is
        # the TARGET's hidden at the last position, which is what the
        # draft's ``fc`` consumes, NOT the draft's mamba2/MLP output. Wire
        # the catchup output into last_hidden so the first proposal step
        # sees the same input as the standalone reference.
        if (
            out_hidden_from_catchup is not None
            and out_hidden_from_catchup.shape == last_hidden.shape
        ):
            last_hidden = out_hidden_from_catchup.clone()

        # --- Initial step: forward on the verified token --------------
        # NOTE: cache_position starts at 1 (not 0) to force HF Mamba2's
        # decode/cache-update branch from the first call. cache_position == 0
        # triggers the SSD prefill path which materializes a (B, n_heads,
        # chunk_size, chunk_size, head_dim) tensor — ~64 GiB at B=8 on Blackwell
        # without the causal_conv1d/selective_state_update C kernels.
        # Cache is zero-initialized (or pulled from caught-up stable cache).
        using_stable = (
            self._current_req_ids is not None
            and len(self._current_req_ids) == B
        )
        if using_stable:
            cache = self._build_batched_stable_cache(
                self._current_req_ids, device, last_hidden.dtype
            )
        else:
            cache = self._fresh_cache(B, device, last_hidden.dtype)
        cache_position = torch.ones(1, device=device, dtype=torch.long)
        zeros_B = torch.zeros(B, dtype=torch.long, device=device)

        # Chain mode: skip verified forward when catchup already advanced the
        # cache through the bonus AND provided the iter-0 cur_hidden via
        # _iter0_cur_hidden. Otherwise the legacy verified forward runs.
        skip_verif = getattr(self, "_skip_verified_forward", False)
        iter0_cur_hidden = getattr(self, "_iter0_cur_hidden", None)
        self._smtprof.end("setup")
        if skip_verif and iter0_cur_hidden is not None and iter0_cur_hidden.shape == last_hidden.shape:
            out_hidden = iter0_cur_hidden
        else:
            self._smtprof.start("verif_fwd")
            out_hidden, _ = self.model.forward(
                input_ids=next_token_ids.view(-1).to(device),
                positions=zeros_B,
                hidden_states=last_hidden,
                cache_params=cache,
                cache_position=cache_position,
            )                                                              # (B, H)
            self._smtprof.end("verif_fwd")
            if using_stable:
                self._store_batched_stable_cache(cache, self._current_req_ids)
        cache_position = cache_position + 1

        # Step 0: pick top-K children of root per request. Fast path: topk in
        # 32K draft vocab, translate selected indices to target vocab via
        # d2t. Avoids ``compute_logits``' inflate-to-128K.
        self._smtprof.start("iter_0_lm_head")
        beam_scores, beam_token_ids = self.model.compute_target_topk(out_hidden, K)
        self._smtprof.end("iter_0_lm_head")
        # paths[b, k] = chain of token ids walked by beam k for request b.
        paths = beam_token_ids.unsqueeze(-1)                           # (B, K, 1)

        # Chain path: beam_width == 1, emits the greedy chain.
        if K == 1:
            # Snapshot capture: save per-request (conv_states, recurrent_states)
            # at each step of the draft loop. The NEXT step's catchup will use
            # snapshots[accept_len] as the new stable state, advancing only by
            # 1 token (the bonus) instead of (accept_len + 1) tokens.
            snapshots_on = (
                self._current_req_ids is not None
                and len(self._current_req_ids) == B
            )
            if snapshots_on:
                # Pre-allocate per-request snapshot buffers if not already
                # there. Layout: (depth, num_layers, 1, ...). Indexed by
                # iter index 0..depth-1. copy_() into these instead of
                # cloning fresh tensors each propose. self._draft_snapshots
                # holds views into these buffers.
                self._draft_snapshots = {}
                for _b, _rid in enumerate(self._current_req_ids):
                    buf = self._snapshot_buf.get(_rid)
                    if buf is None or buf["conv"].shape[0] < depth:
                        # cache slices: (num_layers, 1, channels, kernel)
                        # We need `depth` slots (one per loop entry + each iter
                        # advance; index 0..depth-1).
                        conv_slot = cache.conv_states[:, _b:_b + 1]
                        ssm_slot = cache.ssm_states[:, _b:_b + 1]
                        buf = {
                            "conv": torch.empty(
                                (depth,) + tuple(conv_slot.shape),
                                device=conv_slot.device,
                                dtype=conv_slot.dtype,
                            ),
                            "ssm": torch.empty(
                                (depth,) + tuple(ssm_slot.shape),
                                device=ssm_slot.device,
                                dtype=ssm_slot.dtype,
                            ),
                        }
                        self._snapshot_buf[_rid] = buf
                    # Capture state at loop entry into slot 0.
                    buf["conv"][0].copy_(
                        cache.conv_states[:, _b:_b + 1], non_blocking=True,
                    )
                    buf["ssm"][0].copy_(
                        cache.ssm_states[:, _b:_b + 1], non_blocking=True,
                    )
                    # views, not clones — read-only consumers in next propose.
                    self._draft_snapshots[_rid] = [
                        (buf["conv"][0], buf["ssm"][0])
                    ]

            for _iter_idx in range(depth - 1):
                _iter_n = _iter_idx + 1  # 1..depth-1 (iter_0 already done above)
                cur_ids_flat = paths[..., -1].view(-1)                 # (B,)
                cache_position = cache_position + 1
                self._smtprof.start(f"iter_{_iter_n}_mamba_fwd")
                out_hidden, _ = self.model.forward(
                    input_ids=cur_ids_flat,
                    positions=zeros_B,
                    hidden_states=out_hidden,
                    cache_params=cache,
                    cache_position=cache_position,
                )                                                      # (B, H)
                self._smtprof.end(f"iter_{_iter_n}_mamba_fwd")
                if snapshots_on:
                    # snapshot[i+1] = state after i+1 drafts have been advanced
                    # (the current iter advances by 1 draft). copy_() into
                    # slot _iter_n of the pre-allocated buffer.
                    for _b, _rid in enumerate(self._current_req_ids):
                        buf = self._snapshot_buf[_rid]
                        buf["conv"][_iter_n].copy_(
                            cache.conv_states[:, _b:_b + 1],
                            non_blocking=True,
                        )
                        buf["ssm"][_iter_n].copy_(
                            cache.ssm_states[:, _b:_b + 1],
                            non_blocking=True,
                        )
                        self._draft_snapshots[_rid].append(
                            (buf["conv"][_iter_n], buf["ssm"][_iter_n])
                        )
                self._smtprof.start(f"iter_{_iter_n}_lm_head")
                # Fast greedy: argmax in 32K draft vocab + d2t translate.
                next_id = self.model.compute_target_argmax(out_hidden)
                paths = torch.cat([paths, next_id.view(B, 1, 1)], dim=-1)
                self._smtprof.end(f"iter_{_iter_n}_lm_head")
            out = paths[:, 0, :]
            self._smtprof.end("propose_total")
            self._smtprof.flush()
            return out

        # --- Slow path: real beam search ------------------------------
        # Expand cache to B*K so each beam carries its own Mamba state.
        cache = self._expand_cache(cache, K)
        # Each beam's "parent hidden state" is the same root out_hidden,
        # repeated K times (interleaved so layout is [r0b0, r0b1, ..., r1b0, ...]).
        cur_hidden = out_hidden.repeat_interleave(K, dim=0)            # (B*K, H)
        cur_ids_flat = beam_token_ids.view(-1)                         # (B*K,)
        zeros_BK = torch.zeros(B * K, dtype=torch.long, device=device)

        b_idx = torch.arange(B, device=device).unsqueeze(-1)           # (B, 1)

        for step in range(1, depth):
            cache_position = cache_position + 1
            out_hidden, _ = self.model.forward(
                input_ids=cur_ids_flat,
                positions=zeros_BK,
                hidden_states=cur_hidden,
                cache_params=cache,
                cache_position=cache_position,
            )                                                          # (B*K, H)
            log_probs = torch.log_softmax(
                self.model.compute_logits(out_hidden).float(), dim=-1,
            )                                                          # (B*K, V)
            child = torch.topk(log_probs, K, dim=-1)
            child_lp = child.values.view(B, K, K)                      # (B, K_parent, K_child)
            child_id = child.indices.view(B, K, K)                     # (B, K_parent, K_child)

            cu_scores = (beam_scores.unsqueeze(-1) + child_lp).view(B, K * K)
            winners = torch.topk(cu_scores, K, dim=-1)
            beam_scores = winners.values                               # (B, K)
            flat_idx = winners.indices                                 # (B, K) in [0, K*K)
            parent_idx = flat_idx // K                                 # (B, K)
            child_idx = flat_idx % K                                   # (B, K)
            winner_tokens = child_id[b_idx, parent_idx, child_idx]     # (B, K)

            # Reindex Mamba cache: new beam (b, k_new) inherits state from
            # old beam (b, parent_idx[b, k_new]). In flat layout that's
            # b*K + parent_idx[b, k_new].
            reindex = (b_idx * K + parent_idx).view(-1)                # (B*K,)
            cache = self._reindex_cache(cache, reindex)

            # Carry parent's hidden state into next forward; the next forward
            # will advance it by processing winner_tokens.
            cur_hidden = out_hidden[reindex]                           # (B*K, H)
            cur_ids_flat = winner_tokens.view(-1)                      # (B*K,)

            # Extend paths.
            new_parent_paths = torch.gather(
                paths, dim=1,
                index=parent_idx.unsqueeze(-1).expand(-1, -1, paths.shape[-1]),
            )                                                          # (B, K, step)
            paths = torch.cat([new_parent_paths, winner_tokens.unsqueeze(-1)], dim=-1)

        # Best beam = beam 0 (topk returns sorted descending).
        out = paths[:, 0, :]                                           # (B, depth)
        if logger.isEnabledFor(10):  # DEBUG
            logger.debug(
                "[smeargle] propose: B=%d, depth=%d, K=%d, best_path[:2]=%s",
                B, depth, K, out[:2].tolist(),
            )
        return out






    def _draft_vllm_config(self) -> VllmConfig:
        """Build the VllmConfig get_model wants for the draft.

        Mirrors the minimum that SpecDecodeBaseProposer._create_draft_vllm_config
        does — copy the global config and swap in the draft's ModelConfig.
        """
        from vllm.config import replace
        cfg = self.vllm_config
        return replace(cfg, model_config=cfg.speculative_config.draft_model_config)

    def _fresh_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return Mamba2Cache(
            self._mamba2_config,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def _ensure_pinned_token_buf(self, min_size: int) -> torch.Tensor:
        """Lazy-alloc'd pinned-CPU int64 buffer for staging per-request
        token ids (e.g., snapshot path's bonus_tokens) before async H2D
        transfer. Avoids the synchronous CPU→GPU copy of
        ``torch.tensor(list, device=cuda)`` which serializes the propose
        loop on every call. Pinned memory + non_blocking=True allows the
        H2D copy to overlap with subsequent GPU work.
        """
        buf = getattr(self, "_pinned_token_buf", None)
        if buf is None or buf.shape[0] < min_size:
            new_size = max(min_size, 64)
            new_size = int(new_size * 1.5)
            self._pinned_token_buf = torch.empty(
                new_size, dtype=torch.long, pin_memory=True,
            )
            buf = self._pinned_token_buf
        return buf

    def _ensure_batched_cache_pool(
        self, min_size: int, device: torch.device, dtype: torch.dtype,
    ) -> "Mamba2Cache":
        """Lazy-allocated, reusable Mamba2Cache shared across all batched
        paths in catchup. Saves the per-propose `_fresh_cache(G, ...)` zero
        allocations (~2 MB conv + 16 MB ssm at G=32) which would otherwise
        be allocated and immediately overwritten by torch.cat outputs.

        Grows on demand if a larger G arrives, keeping headroom of 1.5x to
        amortize future regrows.
        """
        pool = getattr(self, "_batched_cache_pool", None)
        if pool is None or pool.conv_states.shape[1] < min_size:
            new_size = max(min_size, 64)
            new_size = int(new_size * 1.5)  # headroom
            self._batched_cache_pool = self._fresh_cache(
                new_size, device, dtype,
            )
            pool = self._batched_cache_pool
        return pool

    def _make_pool_view_cache(
        self, pool: "Mamba2Cache", G: int,
    ) -> "Mamba2Cache":
        """Construct a Mamba2Cache shell whose conv_states/ssm_states are
        views into ``pool.{conv,ssm}_states[:, :G]``. Writes via the shell's
        _LayerView setters propagate to the pool (and are visible to the next
        per-request copy_back). _has_prev/_initialized are owned by the
        shell so multiple successive batched paths within one propose call
        don't pollute each other.
        """
        shell = Mamba2Cache.__new__(Mamba2Cache)
        shell.dtype = pool.dtype
        shell.conv_kernel_size = pool.conv_kernel_size
        shell.n_groups = pool.n_groups
        shell.state_size = pool.state_size
        shell.num_heads = pool.num_heads
        shell.head_dim = pool.head_dim
        shell.intermediate_size = pool.intermediate_size
        shell.conv_states = pool.conv_states[:, :G]
        shell.ssm_states = pool.ssm_states[:, :G]
        n_layers = pool.conv_states.shape[0]
        shell._has_prev = [False] * n_layers
        shell._initialized = [False] * n_layers
        shell.layers = [_LayerView(shell, i) for i in range(n_layers)]
        return shell

    def _expand_cache(self, cache, n: int):
        """Repeat each batch entry n times along the cache batch dim.

        Mamba2Cache stores tensors as ``[num_layers, batch_size, ...]`` so the
        batch dimension is index 1 (matches the reference's
        ``expand_mamba_cache`` helper).
        """
        cache.conv_states = cache.conv_states.repeat_interleave(n, dim=1).contiguous()
        cache.ssm_states = cache.ssm_states.repeat_interleave(n, dim=1).contiguous()
        return cache

    def _reindex_cache(self, cache, indices: torch.Tensor):
        """Permute the batch dim of ``cache`` by ``indices`` (one entry per
        target slot). Mirrors the reference's ``reindex_mamba_cache``.
        """
        cache.conv_states = cache.conv_states[:, indices].contiguous()
        cache.ssm_states = cache.ssm_states[:, indices].contiguous()
        return cache

    def _repeat_interleave_cache(self, cache, counts: torch.Tensor):
        """Repeat each batch entry by per-entry counts. counts is a
        long tensor of shape ``(current_batch,)``. Counts of 0 drop that
        entry. Used by the non-uniform-branching tree walk: each parent
        path at depth d has ``counts[i]`` children at depth d+1, so its
        cache state must be replicated that many times.
        """
        counts = counts.to(cache.conv_states.device)
        cache.conv_states = cache.conv_states.repeat_interleave(
            counts, dim=1
        ).contiguous()
        cache.ssm_states = cache.ssm_states.repeat_interleave(
            counts, dim=1
        ).contiguous()
        return cache

    # ------------------------------------------------------------------
    # Stable per-request Mamba2 cache (mirrors SMEARGLE reference's
    # ``stable_cache_params``). Each request keeps its own batch_size=1
    # cache; we batch-concat at propose() time and split back when storing.
    # ------------------------------------------------------------------

    def _build_batched_stable_cache(
        self,
        req_ids: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Concat per-request caches into a (n_layers, B, ...) batched cache.

        Creates fresh caches lazily for first-seen request ids.
        """
        conv_list: list[torch.Tensor] = []
        ssm_list: list[torch.Tensor] = []
        for r in req_ids:
            c = self.stable_caches.get(r)
            if c is None:
                c = self._fresh_cache(1, device, dtype)
                self.stable_caches[r] = c
            conv_list.append(c.conv_states)
            ssm_list.append(c.ssm_states)
        batched = self._fresh_cache(len(req_ids), device, dtype)
        batched.conv_states = torch.cat(conv_list, dim=1).contiguous()
        batched.ssm_states = torch.cat(ssm_list, dim=1).contiguous()
        # Per-request caches were already prefilled before being stashed in
        # self.stable_caches; propagate the per-layer prefill flag so HF's
        # v5 Mamba2Block takes the decode path on the next forward.
        for i in range(len(batched._has_prev)):
            batched._has_prev[i] = True
            batched._initialized[i] = True
        return batched

    def _store_batched_stable_cache(
        self,
        batched_cache,
        req_ids: list[str],
    ) -> None:
        """Split a batched cache back into per-request caches and save."""
        for b, r in enumerate(req_ids):
            c = self.stable_caches.get(r)
            if c is None:
                # Should have been created in _build_batched_stable_cache.
                continue
            c.conv_states = batched_cache.conv_states[:, b:b + 1].clone()
            c.ssm_states = batched_cache.ssm_states[:, b:b + 1].clone()
            # Inherit prefill flags from the batched cache (split entries
            # share its layer state).
            for i in range(len(c._has_prev)):
                c._has_prev[i] = batched_cache._has_prev[i]
                c._initialized[i] = batched_cache._initialized[i]

    @torch.inference_mode()
    def _maybe_prefill_new_requests(
        self,
        fused: torch.Tensor,
        qsl: torch.Tensor,
        device: torch.device,
        next_token_ids: torch.Tensor | None = None,
    ) -> dict:
        """One-time prefill of a new request's stable cache from its full
        current input slice. Mirrors the reference's "stable_cache_params is
        None" branch in topK_genrate which runs the prompt through the draft
        before any spec decode happens. We do it per-request sequentially
        — the input slice is typically short (single-step prefill), and
        the cost is paid once per request lifetime.
        """
        prefilled_last_hidden: dict[int, torch.Tensor] = {}
        if not self._current_req_ids or self.runner is None:
            return prefilled_last_hidden
        input_ids_gpu = self.runner.input_ids.gpu

        # Identify new requests up front so we know whether to even pay the
        # GPU-sync cost of iterating qsl items.
        new_req_indices: list[tuple[int, str]] = []
        for b, req_id in enumerate(self._current_req_ids):
            if req_id not in self.stable_caches:
                new_req_indices.append((b, req_id))
        if not new_req_indices:
            return prefilled_last_hidden

        qsl_cpu = qsl.cpu().tolist()
        cache_position = torch.ones(1, device=device, dtype=torch.long)
        zeros_1 = torch.zeros(1, dtype=torch.long, device=device)

        # Batched prefill path: group new reqs by exact prompt length T and
        # run one batched model.forward per group instead of B sequential
        # per-request forwards. Big win at start-of-run bursts where many
        # reqs first reach decode together — at B=64 the startup burst is
        # ~63 sequential forwards (~140 ms total) without batching, ~7 groups
        # × ~4 ms = ~28 ms with batching.
        from collections import defaultdict
        by_T: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
        for b, req_id in new_req_indices:
            start = qsl_cpu[b]
            end = qsl_cpu[b + 1]
            if end <= start:
                self.stable_caches[req_id] = self._fresh_cache(
                    1, device, fused.dtype,
                )
                continue
            T_b = end - start
            by_T[T_b].append((b, req_id, start, end))

        for T, group in by_T.items():
            G = len(group)
            if G == 0:
                continue
            ids_list = [
                input_ids_gpu[s:e].long() for (_, _, s, e) in group
            ]
            hid_list = [fused[s:e] for (_, _, s, e) in group]
            batched_ids = torch.stack(ids_list, dim=0)  # (G, T)
            batched_hid = torch.stack(hid_list, dim=0).view(G, T, -1)
            cache_position_full = torch.arange(
                0, T, device=device, dtype=torch.long,
            )
            positions = torch.zeros(T, dtype=torch.long, device=device)
            cache_for_group = self._fresh_cache(G, device, fused.dtype)
            out_hidden, _ = self.model.forward(
                input_ids=batched_ids,
                positions=positions,
                hidden_states=batched_hid,
                cache_params=cache_for_group,
                cache_position=cache_position_full,
            )
            for g, (b, req_id, s, e) in enumerate(group):
                prior_cache = self._fresh_cache(1, device, fused.dtype)
                prior_cache.conv_states.copy_(
                    cache_for_group.conv_states[:, g:g + 1]
                )
                prior_cache.ssm_states.copy_(
                    cache_for_group.ssm_states[:, g:g + 1]
                )
                for i in range(len(prior_cache._has_prev)):
                    prior_cache._has_prev[i] = cache_for_group._has_prev[i]
                    prior_cache._initialized[i] = cache_for_group._initialized[i]
                self.stable_caches[req_id] = prior_cache
                if out_hidden.dim() == 3:
                    prefilled_last_hidden[b] = out_hidden[g, -1]
                else:
                    prefilled_last_hidden[b] = out_hidden[g]
        return prefilled_last_hidden

    def evict_stable_caches(self, keep_req_ids: set[str]) -> None:
        """Drop caches for request_ids not in ``keep_req_ids``. The runner
        can call this periodically to bound memory."""
        stale = [r for r in self.stable_caches if r not in keep_req_ids]
        for r in stale:
            del self.stable_caches[r]
        for r in list(self._cumul_input_ids):
            if r not in keep_req_ids:
                self._cumul_input_ids.pop(r, None)
                self._cumul_hiddens.pop(r, None)
                self._prev_cumul_lens.pop(r, None)
                self._draft_snapshots.pop(r, None)
                self._snapshot_buf.pop(r, None)
                self._prev_qsl_per_rid.pop(r, None)

    @torch.inference_mode()
    def _full_refresh_catch_up(
        self,
        fused: torch.Tensor,
        qsl: torch.Tensor,
        B: int,
        device: torch.device,
        next_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Rebuild stable cache via full SSD prefill over the cumulative
        committed sequence. Matches standalone's `topK_genrate` behavior
        when stable_cache_params is None (else-branch at cnets.py:442-456),
        applied at EVERY step.

        Memory: O(L_max * H) per request. Compute: one SSD prefill per
        step over the entire history — quadratic in total tokens. Use for
        debugging / correctness comparison, not production throughput.
        """
        assert self.runner is not None
        if not self._current_req_ids:
            return None

        # Step 1: extend cumul with THIS step's sampler commits (= step k's
        # accept_lens drafts + step k's new bonus). _bookkeeping_sync runs
        # AFTER propose, so num_tokens_no_spec at propose entry doesn't
        # reflect step k's commits — we read accept_lens directly from
        # _latest_sampled_token_ids stashed by the runner.
        sampled = self._latest_sampled_token_ids
        verified_pos = qsl[:-1].long()
        input_ids_gpu = self.runner.input_ids.gpu

        # Batch all GPU→CPU syncs at function entry. The per-request loop
        # below previously called .item() ~3+accept_lens times per request,
        # each forcing a device sync. At B=64 that was ~200 syncs * ~10-30μs
        # = 2-6 ms per propose. Now: 3 one-shot transfers, no per-loop syncs.
        verified_pos_list = verified_pos.cpu().tolist()
        sampled_list: list[list[int]] | None = (
            sampled.cpu().tolist() if sampled is not None else None
        )
        qsl_total = int(qsl[-1].item())
        # Slice to current step's used range; cheap because input_ids buffer
        # is small (max_num_tokens). Avoids per-token .item() in the loop.
        input_ids_list = input_ids_gpu[:qsl_total].cpu().tolist()

        # Stash per-request accept_lens so the rebuild loop can re-use without
        # recomputing from sampled. Used by both incremental and snapshot paths.
        accept_lens_per_rid: dict[str, int] = {}
        input_ids_len = len(input_ids_list)
        fused_shape0 = fused.shape[0]

        for b, rid in enumerate(self._current_req_ids):
            if rid not in self._cumul_input_ids:
                continue
            cv = verified_pos_list[b]
            # Derive accept_lens from sampled_token_ids: count of valid
            # (non-(-1)) entries minus 1 (the last is the bonus).
            if sampled_list is not None and b < len(sampled_list):
                row = sampled_list[b]
                n_valid = 0
                for v in row:
                    if v >= 0:
                        n_valid += 1
                accept_lens_b = max(0, n_valid - 1)
            else:
                accept_lens_b = 0
            accept_lens_per_rid[rid] = accept_lens_b
            # Per-request prev slot count. If the previous step had more
            # than K+1 tokens for THIS request, it was a prefill step for
            # this req and cumul was already initialized at line ~1109 — do
            # not double-extend.
            prev_per_rid = self._prev_qsl_per_rid.get(rid)
            if (
                prev_per_rid is None
                or prev_per_rid > self.num_speculative_tokens + 1
            ):
                continue  # just-prefilled or new — cumul already in sync

            new_tokens: list[int] = []
            new_hiddens: list[torch.Tensor] = []
            for d in range(accept_lens_b):
                pos = cv + 1 + d
                if pos >= input_ids_len or pos >= fused_shape0:
                    break
                new_tokens.append(input_ids_list[pos])
                new_hiddens.append(fused[pos:pos + 1])
            if (
                sampled_list is not None
                and b < len(sampled_list)
                and accept_lens_b < len(sampled_list[b])
            ):
                bonus_idx = sampled_list[b][accept_lens_b]
                if bonus_idx >= 0:
                    bonus_hidden_pos = cv + accept_lens_b
                    if bonus_hidden_pos < fused_shape0:
                        new_tokens.append(bonus_idx)
                        new_hiddens.append(
                            fused[bonus_hidden_pos:bonus_hidden_pos + 1]
                        )
            if new_tokens:
                self._cumul_input_ids[rid].extend(new_tokens)
                self._cumul_hiddens[rid] = torch.cat(
                    [self._cumul_hiddens[rid]] + new_hiddens, dim=0,
                )

        # Step 2: rebuild stable cache for each request. Each request is
        # routed into one of three batched paths based on its state:
        #   1. **batched snapshot** — request has a per-iter snapshot from the
        #      previous propose() (i.e., it ran a draft loop last step) and
        #      the newly-committed token count matches ``accept_lens + 1``.
        #      We restore the snapshot at index ``accept_lens`` and advance by
        #      one bonus token in a single batched seq_len=1 forward.
        #   2. **batched incremental** — request has a prior cache state but
        #      no usable snapshot (e.g., the verified token count differs).
        #      Feed just the new tokens through the chunk-scan, seeded from
        #      the prior cache.
        #   3. **batched full-refresh** — request has no usable prior state
        #      yet (first decode after prefill on a re-joined request).
        #      Rebuild the cache from zero by replaying its cumulative
        #      committed sequence.
        # Per-request fallback handles the few requests that don't fit any
        # group. Grouping by exact T (or T_new) reduces O(B) kernel launches
        # to O(num_groups), which is usually 1-3.
        snapshots_on = True
        last_out_list: list[torch.Tensor | None] = [None] * len(
            self._current_req_ids
        )

        # --- Single-pass eligibility scan ---------------------------------
        # Combines the three former pre-scans (snapshot/incremental/full-
        # refresh) into one loop over current reqs. Each req lands in
        # exactly one bucket (snapshot > incremental > full-refresh >
        # per-request fallback). Cuts dict lookups 3x.
        from collections import defaultdict
        snapshot_batch: list[tuple[int, str, int, int]] = []
        inc_groups: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
        fr_groups: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
        nspec_p1 = self.num_speculative_tokens + 1
        for b, rid in enumerate(self._current_req_ids):
            cumul_ids = self._cumul_input_ids.get(rid)
            cumul_h = self._cumul_hiddens.get(rid)
            if cumul_ids is None or cumul_h is None or len(cumul_ids) < 2:
                continue
            T = len(cumul_ids) - 1
            prior_cache = self.stable_caches.get(rid)
            prev_T = self._prev_cumul_lens.get(rid)
            accept_lens_b = accept_lens_per_rid.get(rid, 0)
            req_snapshots = self._draft_snapshots.get(rid)
            # 1) Try snapshot
            if (
                prior_cache is not None
                and prev_T is not None
                and prev_T < T
                and req_snapshots is not None
                and 0 <= accept_lens_b < len(req_snapshots)
                and (T - prev_T) == accept_lens_b + 1
            ):
                snapshot_batch.append((b, rid, accept_lens_b, T))
                continue
            # 2) Try incremental
            if (
                prior_cache is not None
                and prev_T is not None
                and prev_T < T
                and prior_cache.has_previous_state(0)
            ):
                inc_groups[T - prev_T].append((b, rid, T))
                continue
            # 3) Try batched full-refresh
            fr_groups[T].append((b, rid, T))

        snapshot_handled_b: set[int] = set()
        incremental_handled_b: set[int] = set()
        full_refresh_handled_b: set[int] = set()

        # ---- Execute Pass 1: batched snapshot ----------------------------
        if snapshot_batch:
            G = len(snapshot_batch)
            conv_list = []
            ssm_list = []
            bonus_tokens = []
            bonus_hiddens = []
            for (b, rid, accept_lens_b, T) in snapshot_batch:
                snap_conv, snap_rec = self._draft_snapshots[rid][accept_lens_b]
                conv_list.append(snap_conv)
                ssm_list.append(snap_rec)
                bonus_tokens.append(self._cumul_input_ids[rid][-1])
                bonus_hiddens.append(self._cumul_hiddens[rid][-1:])
                snapshot_handled_b.add(b)

            pool = self._ensure_batched_cache_pool(G, device, fused.dtype)
            torch.cat(conv_list, dim=1, out=pool.conv_states[:, :G])
            torch.cat(ssm_list, dim=1, out=pool.ssm_states[:, :G])
            batched_cache = self._make_pool_view_cache(pool, G)
            for i in range(len(batched_cache._has_prev)):
                batched_cache._has_prev[i] = True
                batched_cache._initialized[i] = True

            pinned = self._ensure_pinned_token_buf(G)
            for _g, _tok in enumerate(bonus_tokens):
                pinned[_g] = _tok
            batched_input_ids = pinned[:G].to(
                device=device, non_blocking=True,
            ).view(G, 1)
            batched_hidden = torch.cat(bonus_hiddens, dim=0).view(
                G, 1, -1,
            ).to(device=device, dtype=fused.dtype)
            cache_position = torch.ones(1, device=device, dtype=torch.long)
            positions = torch.zeros(G, dtype=torch.long, device=device)

            out_hidden_batched, _ = self.model.forward(
                input_ids=batched_input_ids,
                positions=positions,
                hidden_states=batched_hidden,
                cache_params=batched_cache,
                cache_position=cache_position,
            )
            for g, (b, rid, accept_lens_b, T) in enumerate(snapshot_batch):
                prior_cache = self.stable_caches[rid]
                prior_cache.conv_states.copy_(
                    batched_cache.conv_states[:, g:g + 1]
                )
                prior_cache.ssm_states.copy_(
                    batched_cache.ssm_states[:, g:g + 1]
                )
                for i in range(len(prior_cache._has_prev)):
                    prior_cache._has_prev[i] = batched_cache._has_prev[i]
                    prior_cache._initialized[i] = batched_cache._initialized[i]
                self._prev_cumul_lens[rid] = T
                if out_hidden_batched.dim() == 3:
                    last_out_list[b] = out_hidden_batched[g, -1]
                else:
                    last_out_list[b] = out_hidden_batched[g]
            if not getattr(self, "_batched_snap_logged", False):
                logger.info(
                    "[smeargle] batched snapshot catchup active (G=%d)", G,
                )
                self._batched_snap_logged = True

        # ---- Execute Pass 1.5: batched incremental -----------------------
        for T_new, group in inc_groups.items():
            G = len(group)
            if G == 0:
                continue
            conv_list = []
            ssm_list = []
            ids_list = []
            hid_list = []
            for (b, rid, T) in group:
                prior_cache = self.stable_caches[rid]
                conv_list.append(prior_cache.conv_states)
                ssm_list.append(prior_cache.ssm_states)
                prev_T = T - T_new
                cumul_ids = self._cumul_input_ids[rid]
                cumul_h = self._cumul_hiddens[rid]
                ids_list.append(torch.tensor(
                    cumul_ids[1 + prev_T:1 + prev_T + T_new],
                    dtype=torch.long, device=device,
                ))
                hid_list.append(cumul_h[prev_T:prev_T + T_new])

            pool = self._ensure_batched_cache_pool(G, device, fused.dtype)
            torch.cat(conv_list, dim=1, out=pool.conv_states[:, :G])
            torch.cat(ssm_list, dim=1, out=pool.ssm_states[:, :G])
            batched_cache_inc = self._make_pool_view_cache(pool, G)
            for i in range(len(batched_cache_inc._has_prev)):
                batched_cache_inc._has_prev[i] = True
                batched_cache_inc._initialized[i] = True

            batched_ids = torch.stack(ids_list, dim=0)
            batched_hid = torch.stack(hid_list, dim=0).to(
                device=device, dtype=fused.dtype,
            )
            cache_position = torch.arange(
                0, T_new, device=device, dtype=torch.long,
            )
            positions = torch.zeros(T_new, dtype=torch.long, device=device)

            out_hidden_inc, _ = self.model.forward(
                input_ids=batched_ids,
                positions=positions,
                hidden_states=batched_hid,
                cache_params=batched_cache_inc,
                cache_position=cache_position,
            )

            for g, (b, rid, T) in enumerate(group):
                prior_cache = self.stable_caches[rid]
                prior_cache.conv_states.copy_(
                    batched_cache_inc.conv_states[:, g:g + 1]
                )
                prior_cache.ssm_states.copy_(
                    batched_cache_inc.ssm_states[:, g:g + 1]
                )
                for i in range(len(prior_cache._has_prev)):
                    prior_cache._has_prev[i] = batched_cache_inc._has_prev[i]
                    prior_cache._initialized[i] = batched_cache_inc._initialized[i]
                self._prev_cumul_lens[rid] = T
                if out_hidden_inc.dim() == 3:
                    last_out_list[b] = out_hidden_inc[g, -1]
                else:
                    last_out_list[b] = out_hidden_inc[g]
                incremental_handled_b.add(b)
        if inc_groups and not getattr(self, "_batched_inc_logged", False):
            logger.info(
                "[smeargle] batched incremental catchup active "
                "(groups=%d, total_reqs=%d)",
                len(inc_groups), sum(len(v) for v in inc_groups.values()),
            )
            self._batched_inc_logged = True

        # ---- Execute Pass 1.75: batched full-refresh ---------------------
        for T, group in fr_groups.items():
            G = len(group)
            if G == 0:
                continue
            ids_list = []
            hid_list = []
            for (b, rid, _T) in group:
                cumul_ids = self._cumul_input_ids[rid]
                cumul_h = self._cumul_hiddens[rid]
                ids_list.append(torch.tensor(
                    cumul_ids[1:1 + T], dtype=torch.long, device=device,
                ))
                hid_list.append(cumul_h[:T])
            batched_ids = torch.stack(ids_list, dim=0)
            batched_hid = torch.stack(hid_list, dim=0).to(
                device=device, dtype=fused.dtype,
            )
            cache_position = torch.arange(
                0, T, device=device, dtype=torch.long,
            )
            positions = torch.zeros(T, dtype=torch.long, device=device)
            pool = self._ensure_batched_cache_pool(G, device, fused.dtype)
            pool.conv_states[:, :G].zero_()
            pool.ssm_states[:, :G].zero_()
            cache_for_group = self._make_pool_view_cache(pool, G)

            out_hidden_fr, _ = self.model.forward(
                input_ids=batched_ids,
                positions=positions,
                hidden_states=batched_hid,
                cache_params=cache_for_group,
                cache_position=cache_position,
            )

            for g, (b, rid, _T) in enumerate(group):
                prior_cache = self._fresh_cache(1, device, fused.dtype)
                prior_cache.conv_states.copy_(
                    cache_for_group.conv_states[:, g:g + 1]
                )
                prior_cache.ssm_states.copy_(
                    cache_for_group.ssm_states[:, g:g + 1]
                )
                for i in range(len(prior_cache._has_prev)):
                    prior_cache._has_prev[i] = cache_for_group._has_prev[i]
                    prior_cache._initialized[i] = cache_for_group._initialized[i]
                self.stable_caches[rid] = prior_cache
                self._prev_cumul_lens[rid] = T
                if out_hidden_fr.dim() == 3:
                    last_out_list[b] = out_hidden_fr[g, -1]
                else:
                    last_out_list[b] = out_hidden_fr[g]
                full_refresh_handled_b.add(b)
        if fr_groups and not getattr(self, "_batched_fr_logged", False):
            logger.info(
                "[smeargle] batched full-refresh active "
                "(groups=%d, total_reqs=%d, T_dist=%s)",
                len(fr_groups),
                sum(len(v) for v in fr_groups.values()),
                {T: len(g) for T, g in fr_groups.items()},
            )
            self._batched_fr_logged = True

        # --- Pass 2: per-request loop for the remainder -------------------
        for b, rid in enumerate(self._current_req_ids):
            if (
                b in snapshot_handled_b
                or b in incremental_handled_b
                or b in full_refresh_handled_b
            ):
                continue  # already handled in batched passes
            cumul_ids = self._cumul_input_ids.get(rid)
            cumul_h = self._cumul_hiddens.get(rid)
            if cumul_ids is None or cumul_h is None or len(cumul_ids) < 2:
                last_out_list[b] = fused.new_zeros(fused.shape[-1])
                continue
            L = len(cumul_ids)
            T = L - 1  # cumul prefill uses cumul[1:] / cumul_h[:T]

            prior_cache = self.stable_caches.get(rid)
            prev_T = self._prev_cumul_lens.get(rid)
            accept_lens_b = accept_lens_per_rid.get(rid, 0)
            req_snapshots = self._draft_snapshots.get(rid)
            use_snapshot = (
                snapshots_on
                and prior_cache is not None
                and prev_T is not None
                and prev_T < T
                and req_snapshots is not None
                and 0 <= accept_lens_b < len(req_snapshots)
                # We only have snapshots for the K=1 chain path; sanity-check
                # that the new tokens added match the expected (accept_lens + 1)
                # = 1 bonus advance. If not, fall through to incremental.
                and (T - prev_T) == accept_lens_b + 1
            )
            use_incremental = (
                not use_snapshot
                and incremental_on
                and prior_cache is not None
                and prev_T is not None
                and prev_T < T
                and prior_cache.has_previous_state(0)
            )

            if use_snapshot:
                # Restore the cache to the snapshot taken at iter `accept_lens_b`
                # of the previous propose's draft loop, then advance by 1 token
                # (the new bonus). This eliminates the (accept_lens_b)-token
                # catchup prefill — only the bonus single-token forward remains.
                snap_conv, snap_rec = req_snapshots[accept_lens_b]
                prior_cache.conv_states.copy_(snap_conv)
                prior_cache.ssm_states.copy_(snap_rec)
                # has_previous_state[0] should be True (snapshot taken after
                # at least the verified_forward populated the cache); leave it.
                bonus_token = cumul_ids[-1]
                bonus_hidden = cumul_h[-1:].view(1, 1, -1).to(
                    device=device, dtype=fused.dtype,
                )
                prompt_ids_gpu = torch.tensor(
                    [bonus_token], dtype=torch.long, device=device,
                ).view(1, 1)
                cache_position_full = torch.tensor(
                    [prev_T + accept_lens_b], dtype=torch.long, device=device,
                )
                positions = torch.zeros(1, dtype=torch.long, device=device)
                cache = prior_cache
                out_hidden, _ = self.model.forward(
                    input_ids=prompt_ids_gpu,
                    positions=positions,
                    hidden_states=bonus_hidden,
                    cache_params=cache,
                    cache_position=cache_position_full,
                )
                if not getattr(self, "_snapshot_logged", False):
                    logger.info(
                        "[smeargle] snapshot-based catchup active "
                        "(accept_len=%d, bonus-only advance)", accept_lens_b,
                    )
                    self._snapshot_logged = True
            elif use_incremental:
                T_new = T - prev_T
                # Feed JUST the new tokens, seeded from prior_cache state.
                prompt_ids_gpu = torch.tensor(
                    cumul_ids[1 + prev_T:1 + prev_T + T_new],
                    dtype=torch.long, device=device,
                ).view(1, T_new)
                prompt_hidden = cumul_h[prev_T:prev_T + T_new].view(
                    1, T_new, -1
                ).to(device=device, dtype=fused.dtype)
                cache_position_full = torch.arange(
                    prev_T, prev_T + T_new, device=device, dtype=torch.long,
                )
                positions = torch.zeros(T_new, dtype=torch.long, device=device)
                cache = prior_cache  # continue from prior state
                out_hidden, _ = self.model.forward(
                    input_ids=prompt_ids_gpu,
                    positions=positions,
                    hidden_states=prompt_hidden,
                    cache_params=cache,
                    cache_position=cache_position_full,
                )
                if not getattr(self, "_incremental_logged", False):
                    logger.info(
                        "[smeargle] incremental chunk catchup active "
                        "(T_new=%d, prev_T=%d, T=%d)",
                        T_new, prev_T, T,
                    )
                    self._incremental_logged = True
            else:
                # Full-refresh path: fresh cache, prefill over full cumul.
                prompt_ids_gpu = torch.tensor(
                    cumul_ids[1:], dtype=torch.long, device=device,
                ).view(1, T)
                prompt_hidden = cumul_h[:T].view(1, T, -1).to(
                    device=device, dtype=fused.dtype,
                )
                cache_position_full = torch.arange(
                    0, T, device=device, dtype=torch.long,
                )
                positions = torch.zeros(T, dtype=torch.long, device=device)
                cache = self._fresh_cache(1, device, fused.dtype)
                out_hidden, _ = self.model.forward(
                    input_ids=prompt_ids_gpu,
                    positions=positions,
                    hidden_states=prompt_hidden,
                    cache_params=cache,
                    cache_position=cache_position_full,
                )

            # Save back to per-request stable cache for the iter loop.
            self.stable_caches[rid] = cache
            # Track current cumul length so the NEXT step knows how many
            # tokens are new.
            self._prev_cumul_lens[rid] = T
            if out_hidden.dim() == 3:
                last_out_list[b] = out_hidden[0, -1]
            else:
                last_out_list[b] = out_hidden[-1]

        # Backfill any None slots (shouldn't happen unless cumul missing) with
        # zero so torch.stack below doesn't blow up.
        for _b in range(len(last_out_list)):
            if last_out_list[_b] is None:
                last_out_list[_b] = fused.new_zeros(fused.shape[-1])
        if not last_out_list:
            return None
        return torch.stack(last_out_list, dim=0).to(device=device, dtype=fused.dtype)



# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _find_embed_tokens(model: nn.Module):
    """Walk a target model to find its embed_tokens, if any."""
    m = model
    for name in ("model", "language_model"):
        if hasattr(m, name):
            sub = getattr(m, name)
            if hasattr(sub, "embed_tokens"):
                return sub.embed_tokens
            m = sub
    return getattr(m, "embed_tokens", None)


class set_default_torch_dtype:
    """Context manager that sets ``torch.set_default_dtype`` for the duration."""

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.prev = None

    def __enter__(self):
        self.prev = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        return self

    def __exit__(self, *exc):
        torch.set_default_dtype(self.prev)
