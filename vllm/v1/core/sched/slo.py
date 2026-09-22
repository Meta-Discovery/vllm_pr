# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SLO-aware scheduling helpers (env-gated, OFF by default).

This module packages three output-exact serving controls behind a single
operator SLO knob. It only affects *when* work is admitted and *how large* a
prefill chunk is per step — never token values — so generations are byte-
identical to the stock scheduler (greedy). With ``VLLMSLOConfig.enabled`` False
(the default) every entry point is a no-op and the scheduler behaves exactly as
gold.

Controls
--------
1. ITL chunk-cap: bound the prefill tokens co-scheduled with decodes each step so
   the predicted step time stays under the tail-ITL SLO, using a measured
   decode-cost roofline ``step_ms ≈ base + k·M**exp`` (sqrt-M; online-calibrated).
2. VTC fairness: per-client virtual-time fair queuing on admission
   (under-served clients admit first; weighted shares), aging-bounded.
3. Feasibility: detect when the requested SLO is below the load-induced decode
   floor (report), and derive the achievable floor (act, via an external
   governor).

Kept intentionally dependency-free (stdlib only) so the logic is unit-testable
without importing the engine or touching a GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class VLLMSLOConfig:
    """Configuration for the SLO controls. Disabled unless VLLM_SLO_ENABLE=1."""

    enabled: bool = False
    itl_slo_ms: float = 40.0        # tail inter-token-latency SLO (the one operator knob)
    min_prefill: int = 256          # floor on prefill tokens/step (never starve to 0)
    backlog_max: int = 1_000_000    # bypass the cap when the waiting queue exceeds this
    safety: float = 0.9             # target step time = safety * SLO (margin for the tail)
    vtc_enabled: bool = False
    vtc_quantum: float = 4096.0     # per-client credit before deferring an over-served client
    # decode-cost roofline (RUN65: step_ms ≈ base + per_req · M**exp, R²=0.99)
    base_ms: float = 5.9
    per_req_ms: float = 0.49
    decode_exp: float = 0.5
    pref_per_tok_ms: float = 0.010  # per-prefill-token step cost (online-calibrated)

    @classmethod
    def from_env(cls) -> "VLLMSLOConfig":
        return cls(
            enabled=os.environ.get("VLLM_SLO_ENABLE", "0") == "1",
            itl_slo_ms=_envf("VLLM_SLO_ITL_MS", 40.0),
            min_prefill=_envi("VLLM_SLO_MIN_PREFILL", 256),
            vtc_enabled=os.environ.get("VLLM_SLO_VTC", "0") == "1",
        )


class DecodeCostModel:
    """Measured decode-step roofline with online EMA calibration.

    step_ms(M) ≈ base + per_req · M**exp  (sqrt-M when exp=0.5). The prefill term
    is linear in co-scheduled prefill tokens: + pref_per_tok · n_prefill_tokens.
    """

    def __init__(self, cfg: VLLMSLOConfig, ema: float = 0.05) -> None:
        self.base = cfg.base_ms
        self.per_req = cfg.per_req_ms
        self.exp = cfg.decode_exp
        self.pref_per_tok = cfg.pref_per_tok_ms
        self._ema = ema

    def decode_est(self, m: int) -> float:
        """Estimated decode-only step time (ms) for a running batch of M decodes."""
        if m <= 0:
            return self.base
        return self.base + self.per_req * (m ** self.exp)

    def update(self, elapsed_ms: float, n_ctx_tokens: int, n_gen_reqs: int) -> None:
        """Online EMA calibration from an observed step (ignored if implausible)."""
        if elapsed_ms <= 0.0 or elapsed_ms > 5000.0:
            return
        b = self._ema
        if n_ctx_tokens == 0 and n_gen_reqs > 0:
            per = (elapsed_ms - self.base) / (n_gen_reqs ** self.exp)
            if per > 0:
                self.per_req = (1 - b) * self.per_req + b * per
        elif n_ctx_tokens > 0:
            slope = (elapsed_ms - self.decode_est(n_gen_reqs)) / n_ctx_tokens
            if slope > 0:
                self.pref_per_tok = (1 - b) * self.pref_per_tok + b * slope

    def prefill_chunk_cap(self, cfg: VLLMSLOConfig, n_decoders: int,
                          waiting: int = 0) -> int:
        """Max prefill tokens to co-schedule this step to keep step_ms <= safety*SLO.

        Returns a very large number (no cap) when no decode is running or when the
        waiting backlog is deep (avoid diverging TTFT by starving prefill).
        """
        if n_decoders <= 0:
            return 1 << 30
        if waiting > cfg.backlog_max:
            return 1 << 30
        avail = cfg.safety * cfg.itl_slo_ms - self.decode_est(n_decoders)
        cap = avail / max(self.pref_per_tok, 1e-6)
        return max(cfg.min_prefill, int(cap))


def client_key(request_id: str) -> str:
    """Client identity for fair queuing from a request id.

    Ids look like ``cmpl-<client>-<n>`` (the client segment comes from the
    X-Request-Id / --request-id-prefix). Strip a leading ``cmpl-`` then the last
    ``-<suffix>`` so all requests from one client share a key. Falls back to the
    whole id (=> each request is its own client => VTC no-ops) if unstructured.
    """
    s = request_id
    if s.startswith("cmpl-"):
        s = s[len("cmpl-"):]
    i = s.rfind("-")
    return s[:i] if i > 0 else s


class VTCFairness:
    """Virtual-Time fair queuing on admission (Sheng et al., OSDI'24), weighted.

    Tracks tokens served per client; virtual time v_c = served_c / weight_c. On
    admission, defer a client whose v_c exceeds the least-served waiting client's
    frontier by more than one quantum, so under-served clients admit first.
    Non-preemptive, aging-bounded, output-exact (admission ORDER only).
    """

    def __init__(self) -> None:
        self.served: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def set_weights(self, weights: dict[str, float]) -> None:
        if weights != self.weights:
            self.served.clear()  # fresh fair frontier when the policy changes
            self.weights = dict(weights)

    def weight(self, client: str) -> float:
        return self.weights.get(client, 1.0)

    def vtime(self, client: str) -> float:
        return self.served.get(client, 0.0) / self.weight(client)

    def record(self, client: str, tokens: float) -> None:
        self.served[client] = self.served.get(client, 0.0) + tokens

    def frontier(self, waiting_clients) -> float:
        # Include NEVER-served clients at virtual-time 0 (via vtime()'s default):
        # a brand-new light client is maximally under-served and must lower the
        # frontier so an over-served heavy client is deferred (protects newcomers).
        vs = [self.vtime(c) for c in waiting_clients]
        return min(vs) if vs else 0.0

    def should_defer(self, client: str, waiting_clients, quantum: float) -> bool:
        """True if `client` is over-served vs the waiting frontier by > quantum."""
        wc = list(waiting_clients)
        if len(set(wc)) <= 1:
            return False  # single client -> no fairness to enforce
        return self.vtime(client) > self.frontier(wc) + quantum


# --- feasibility helpers (used by the external DVFS/SLO governor) -------------

def windowed_p99(prev_buckets: dict, cur_buckets: dict, q: float = 0.99):
    """Estimate the q-percentile ITL (ms) over a window from cumulative-histogram
    deltas (vLLM ``inter_token_latency_seconds`` buckets). None if too few samples.
    """
    if not prev_buckets or not cur_buckets:
        return None
    inf = float("inf")
    les = sorted(k for k in cur_buckets if k != inf) + [inf]
    deltas = [(le, max(0.0, cur_buckets.get(le, 0.0) - prev_buckets.get(le, 0.0)))
              for le in les]
    total = sum(d for _, d in deltas)
    if total < 5:
        return None
    target = q * total - 1e-9  # tolerate q*total float rounding on exact boundaries
    cum = 0.0
    for i, (le, d) in enumerate(deltas):
        cum += d
        if cum >= target:
            return (le * 1000.0) if le != inf else (les[i - 1] * 1000.0)
    return None


@dataclass
class FeasibilityState:
    """Closed-loop feasibility: relax the effective SLO to the measured floor when
    the requested SLO is unachievable at max clock; re-tighten when load eases.
    Pure state machine (no I/O) so it is unit-testable.
    """

    requested_ms: float
    eff_ms: float = field(default=0.0)
    infeas_run: int = 0
    slack_run: int = 0
    relax_margin: float = 1.10
    infeas_holds: int = 5
    retighten_holds: int = 8
    retighten_frac: float = 0.65
    retighten_frac_step: float = 0.15

    def __post_init__(self) -> None:
        if self.eff_ms <= 0.0:
            self.eff_ms = self.requested_ms

    def feasible(self) -> bool:
        return self.eff_ms <= self.requested_ms + 0.5

    def step(self, pinned_max: bool, tail_ms: float) -> str:
        """Advance one control tick. Returns 'relax' | 'retighten' | 'hold'."""
        if pinned_max and tail_ms > self.eff_ms:
            self.infeas_run += 1
            self.slack_run = 0
            if self.infeas_run >= self.infeas_holds:
                new_eff = max(self.requested_ms, tail_ms * self.relax_margin)
                self.infeas_run = 0
                if new_eff > self.eff_ms + 0.5:
                    self.eff_ms = new_eff
                    return "relax"
        elif (self.eff_ms > self.requested_ms + 0.5 and not pinned_max
              and tail_ms < self.eff_ms * self.retighten_frac):
            self.slack_run += 1
            self.infeas_run = 0
            if self.slack_run >= self.retighten_holds:
                self.slack_run = 0
                self.eff_ms = max(self.requested_ms,
                                  self.eff_ms - self.retighten_frac_step * self.eff_ms)
                return "retighten"
        else:
            self.infeas_run = max(0, self.infeas_run - 1)
            self.slack_run = max(0, self.slack_run - 1)
        return "hold"
