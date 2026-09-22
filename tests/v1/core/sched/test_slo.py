# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the env-gated SLO scheduling controls (vllm/v1/core/sched/slo.py).

Covers the three controls the operator relies on — the ITL prefill chunk-cap, VTC
per-client fairness, and the closed-loop feasibility state machine — plus the
default-off / output-exact contract. Pure logic: no GPU, no model, no engine.
"""
import pytest

from vllm.v1.core.sched.slo import (
    DecodeCostModel,
    FeasibilityState,
    VLLMSLOConfig,
    VTCFairness,
    client_key,
    windowed_p99,
)


# --- client-key extraction (request_id -> client for VTC) ---------------------

def test_client_key_strips_cmpl_prefix_and_suffix():
    assert client_key("cmpl-clientA-0") == "clientA"
    assert client_key("cmpl-clientA-137") == "clientA"


def test_client_key_without_cmpl_prefix():
    assert client_key("clientB-9") == "clientB"


def test_client_key_unstructured_is_own_client():
    # a bare uuid-style id with no client structure -> its own client (VTC no-ops)
    assert client_key("abc123") == "abc123"


# --- config / default-off contract -------------------------------------------

def test_disabled_by_default():
    assert VLLMSLOConfig().enabled is False
    assert VLLMSLOConfig().vtc_enabled is False


def test_from_env_off_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_SLO_ENABLE", raising=False)
    assert VLLMSLOConfig.from_env().enabled is False


def test_from_env_on(monkeypatch):
    monkeypatch.setenv("VLLM_SLO_ENABLE", "1")
    monkeypatch.setenv("VLLM_SLO_ITL_MS", "30")
    cfg = VLLMSLOConfig.from_env()
    assert cfg.enabled is True and cfg.itl_slo_ms == 30.0


# --- decode-cost roofline -----------------------------------------------------

def test_decode_est_sqrt_shape_monotonic():
    m = DecodeCostModel(VLLMSLOConfig())
    assert m.decode_est(0) == pytest.approx(m.base)
    # strictly increasing in M, and sublinear (sqrt): step(4M)-step(M) shrinks per-M
    vals = [m.decode_est(x) for x in (1, 4, 16, 64)]
    assert vals == sorted(vals)
    # sqrt(64)=8 -> base + 8*per_req
    assert m.decode_est(64) == pytest.approx(m.base + m.per_req * 8.0)


def test_cost_model_online_calibration_moves_params():
    m = DecodeCostModel(VLLMSLOConfig())
    before = m.per_req
    # feed pure-decode steps slower than the model predicts -> per_req rises
    for _ in range(50):
        m.update(elapsed_ms=m.base + 5.0, n_ctx_tokens=0, n_gen_reqs=16)
    assert m.per_req > before
    # implausible samples are ignored (no NaN/blowup)
    stable = m.per_req
    m.update(elapsed_ms=-1, n_ctx_tokens=0, n_gen_reqs=16)
    m.update(elapsed_ms=99999, n_ctx_tokens=0, n_gen_reqs=16)
    assert m.per_req == stable


# --- ITL prefill chunk-cap ----------------------------------------------------

def test_prefill_cap_no_decoders_is_uncapped():
    cfg = VLLMSLOConfig(enabled=True)
    m = DecodeCostModel(cfg)
    assert m.prefill_chunk_cap(cfg, n_decoders=0) >= (1 << 29)


def test_prefill_cap_monotonic_in_slo():
    m = DecodeCostModel(VLLMSLOConfig())
    loose = m.prefill_chunk_cap(VLLMSLOConfig(itl_slo_ms=40), n_decoders=32)
    tight = m.prefill_chunk_cap(VLLMSLOConfig(itl_slo_ms=20), n_decoders=32)
    # a looser SLO permits >= as many prefill tokens per step
    assert loose >= tight


def test_prefill_cap_floors_at_min_prefill():
    cfg = VLLMSLOConfig(itl_slo_ms=1.0, min_prefill=256)  # impossibly tight
    m = DecodeCostModel(cfg)
    assert m.prefill_chunk_cap(cfg, n_decoders=64) == 256  # never starves to 0


def test_prefill_cap_bypasses_on_deep_backlog():
    cfg = VLLMSLOConfig(itl_slo_ms=20, backlog_max=100)
    m = DecodeCostModel(cfg)
    capped = m.prefill_chunk_cap(cfg, n_decoders=32, waiting=0)
    bypass = m.prefill_chunk_cap(cfg, n_decoders=32, waiting=101)
    assert capped < (1 << 29) and bypass >= (1 << 29)


# --- VTC fairness -------------------------------------------------------------

def test_vtc_virtual_time_weighted():
    v = VTCFairness()
    v.set_weights({"A": 1.0, "B": 3.0})
    v.record("A", 3000)
    v.record("B", 3000)
    # same tokens, higher weight -> lower virtual time (B is "under-served")
    assert v.vtime("B") < v.vtime("A")


def test_vtc_defers_over_served_admits_under_served():
    v = VTCFairness()
    v.set_weights({"A": 1.0, "B": 1.0})
    v.record("A", 100000)  # A floods
    v.record("B", 10)      # B barely served
    waiting = ["A", "B"]
    assert v.should_defer("A", waiting, quantum=4096) is True   # over-served -> defer
    assert v.should_defer("B", waiting, quantum=4096) is False  # under-served -> admit


def test_vtc_protects_brand_new_light_client():
    """A never-served light client (not yet in `served`) must be recognized as
    under-served (vtime 0) so the flooding client is deferred on its behalf —
    the 'lift'/frontier fix. Without it a newcomer is never prioritized."""
    v = VTCFairness()
    v.record("heavy", 100000)      # heavy has flooded; "new" client never served
    waiting = ["heavy", "new"]     # 'new' not in v.served
    assert v.should_defer("heavy", waiting, quantum=4096) is True
    assert v.should_defer("new", waiting, quantum=4096) is False


def test_vtc_single_client_never_defers():
    v = VTCFairness()
    v.record("A", 100000)
    assert v.should_defer("A", ["A"], quantum=4096) is False


def test_vtc_weights_change_resets_frontier():
    v = VTCFairness()
    v.set_weights({"A": 1.0})
    v.record("A", 5000)
    v.set_weights({"A": 1.0, "B": 2.0})  # policy change
    assert v.served == {}  # counters reset to a fresh fair frontier


# --- feasibility: windowed p99 + closed-loop state machine --------------------

def test_windowed_p99_from_bucket_deltas():
    # cumulative histograms; 100 samples in window, ~99th percentile in the 0.05 bucket
    prev = {0.01: 0, 0.025: 0, 0.05: 0, float("inf"): 0}
    cur = {0.01: 50, 0.025: 90, 0.05: 99, float("inf"): 100}
    assert windowed_p99(prev, cur) == pytest.approx(50.0)  # le=0.05 -> 50 ms


def test_windowed_p99_none_on_too_few_samples():
    prev = {0.05: 0, float("inf"): 0}
    cur = {0.05: 2, float("inf"): 2}
    assert windowed_p99(prev, cur) is None


def test_feasibility_relaxes_when_infeasible():
    fs = FeasibilityState(requested_ms=20.0, infeas_holds=3)
    assert fs.feasible() and fs.eff_ms == 20.0
    acts = [fs.step(pinned_max=True, tail_ms=30.0) for _ in range(3)]
    assert acts[-1] == "relax"
    assert fs.eff_ms > 20.0 and not fs.feasible()  # relaxed to ~floor, reports infeasible


def test_feasibility_retightens_when_load_eases():
    fs = FeasibilityState(requested_ms=20.0, eff_ms=40.0,
                          retighten_holds=3, retighten_frac=0.65)
    # sustained slack (tail well below eff, not pinned) -> re-tighten down toward 20
    acts = [fs.step(pinned_max=False, tail_ms=10.0) for _ in range(3)]
    assert acts[-1] == "retighten"
    assert 20.0 <= fs.eff_ms < 40.0


def test_feasibility_never_tightens_below_requested():
    fs = FeasibilityState(requested_ms=20.0, eff_ms=21.0,
                          retighten_holds=1, retighten_frac=0.9)
    for _ in range(50):
        fs.step(pinned_max=False, tail_ms=1.0)
    assert fs.eff_ms >= 20.0  # clamps at the requested SLO


def test_vtc_admission_integration_under_served_first():
    """Integration of the pieces the scheduler waiting-loop hook composes:
    client_key(request_id) + VTCFairness.should_defer over a mixed waiting queue.
    A heavy client floods; a light client must not be starved — simulate one
    admission pass and assert the light client's request admits, the heavy one
    defers, exactly as the in-tree waiting-loop hook would decide."""
    vtc = VTCFairness()
    vtc.set_weights({"heavy": 1.0, "light": 1.0})
    vtc.record("heavy", 100_000)   # heavy has flooded
    vtc.record("light", 0)         # light barely served
    # waiting queue holds requests from both clients (request-id shaped as served)
    waiting_ids = ["cmpl-heavy-7", "cmpl-light-3"]
    clients = {client_key(r) for r in waiting_ids}
    admitted, deferred = [], []
    for rid in waiting_ids:
        ck = client_key(rid)
        if vtc.should_defer(ck, clients, quantum=4096):
            deferred.append(rid)
        else:
            admitted.append(rid)
    assert "cmpl-light-3" in admitted
    assert "cmpl-heavy-7" in deferred


def test_vtc_distinct_uuid_clients_never_defer():
    """When each request is its own client (cmpl-<uuid>, no shared prefix), VTC
    must no-op — mirrors real cmpl-<uuid> traffic where fairness does nothing."""
    vtc = VTCFairness()
    ids = ["cmpl-abc123-0", "cmpl-def456-0", "cmpl-999aaa-0"]
    for rid in ids:
        vtc.record(client_key(rid), 5000)
    clients = {client_key(r) for r in ids}
    assert all(not vtc.should_defer(client_key(r), clients, 4096) for r in ids)


def test_feasibility_hold_when_on_target():
    fs = FeasibilityState(requested_ms=30.0)
    # at target, not pinned, tail comfortably under -> no change (feasible)
    for _ in range(10):
        assert fs.step(pinned_max=False, tail_ms=20.0) == "hold"
    assert fs.eff_ms == 30.0 and fs.feasible()
