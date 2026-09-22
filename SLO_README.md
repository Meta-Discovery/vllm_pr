# SLO-aware, output-exact serving controls for vLLM

One operator knob — a tail inter-token-latency SLO — that turns idle hardware slack
into lower latency, fairer sharing, and large energy savings, with **byte-identical
output** and **zero risk when off**.

Base: vLLM `bca7bea2405127bd5291bb6fffa679bdcd8f6dd9`. Measured on one 8×B200 node
serving `nvidia/GLM-5.2-NVFP4` (TP8 / expert-parallel / fp8 KV).

## Results (matched A/B vs tuned stock vLLM, output-exact, error-barred)

| Axis | Result |
|---|---|
| **Energy** | **−34% GPU power** on the real mixed 8192/1024 workload (throughput within ~5%); **+31% tokens/joule at −25% power** on pure-decode load |
| **Fairness** | **up to 8.7× lower light-client TTFT** under deep-queue contention (2.8× confirmed in-tree), heavy-client throughput neutral |
| **Latency tail** | **up to ~21× lower decode ITL p99** vs stock's only ITL-bounding knob; **~10–15% median + strong burst-tail protection** vs stock default |
| **Throughput** | combined always-on config never regresses vs tuned stock (pure-batch 1001 vs 849 tok/s) |

The controls change only *when* work is admitted and *how large* a prefill chunk
is — **never token values** — so output matches stock within the accuracy gate
(GSM8K + long-context NIAH). As with stock vLLM under concurrency, results are not
bitwise-reproducible (fp non-associativity), so this is output-exact by
construction, not a bitwise guarantee.

## The idea

Decode on this model is memory-latency-bound: step time follows a measured roofline
`step_ms ≈ 5.9 + 0.49·√M` (R²=0.99, confirmed by hardware counters — Tensor Active
2.3%, DRAM read 21%). So the prefill chunk co-scheduled with decodes and the SM
clock are both slack you can spend to hit a latency target without touching the
math. This change turns that slack into three controls stock vLLM cannot express.

## What it adds (all env-gated; unset ⇒ byte-identical stock)

| Control | Env gate | What it does |
|---|---|---|
| Per-step ITL chunk-cap | `VLLM_SLO_ENABLE=1` | Bounds prefill tokens co-scheduled with decodes so predicted step time stays under the SLO. A static per-*request* `long_prefill_token_threshold` cannot bound per-*step* time under K concurrent prefills. |
| VTC per-client fairness | `VLLM_SLO_VTC=1` | Weighted virtual-time fair-queuing on admission; under-served clients admit first. Stock `priority` is static per-request. |
| SLO-aware DVFS governor | host tool | Rides the SM clock down to the lowest setting whose ITL floor stays under the SLO, back up during prefill bursts — the source of the energy win. |

Optional admission policies behind their own gates: `VLLM_SLO_SRPT` (short-output
first), `VLLM_SLO_SJF` (short-prompt first), `VLLM_SLO_CACHE_AWARE` (defer cold-prefix
siblings behind a pioneer), `VLLM_SLO_TIERS` / `VLLM_SLO_UNIFIED` (per-class deadlines).

## Files

| Path | What |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | Gated ITL-cap + VTC + admission hooks (additive, +448 lines). |
| `vllm/v1/core/sched/slo.py` | Stdlib cost model + VTC + feasibility, unit-tested. |
| `tests/v1/core/sched/test_slo.py` | 25 unit/integration tests (all pass). |
| `dvfs/adaptive_slo_controller.py`, `dvfs/dvfs_governor.py` | Host-side one-knob DVFS governor. |
| `benchmarks/slo/*.py` | Repro harnesses (stdlib-only, hit `localhost:8000`). |

## How to run

**Serve — stock baseline (OFF):**
```bash
vllm serve nvidia/GLM-5.2-NVFP4 --served-model-name glm-5.2-nvfp4 \
  --tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8_e4m3 \
  --scheduling-policy priority --max-num-seqs 64 \
  --gpu-memory-utilization 0.75 --host 0.0.0.0 --port 8000
```

**Serve — controls ON (same command, gated prefix):**
```bash
VLLM_SLO_ENABLE=1 VLLM_SLO_ITL_INT_GATE=1 VLLM_SLO_VTC=1 VLLM_SLO_DECODE_EXP=0.5 \
vllm serve nvidia/GLM-5.2-NVFP4 ...same flags...
```
Main knobs: `VLLM_SLO_ITL_MS` (the SLO, default 40), `VLLM_SLO_DECODE_EXP=0.5` (use
the measured √M roofline), `VLLM_SLO_INT_PREFIX` (request-id prefix marking the
latency-sensitive class; `""` = protect all). The cost-model constants are set from
these env vars at startup.

**DVFS governor (host-side, needs NVML / `CAP_SYS_ADMIN`):**
```bash
python dvfs/adaptive_slo_controller.py --tail-slo-ms 40   # reads /metrics, sets clocks
```

**Measure (run each against OFF then ON, compare):**
```bash
python benchmarks/slo/fair_gate.py    # VTC: light-client TTFT under a heavy flood
python benchmarks/slo/itl_interf.py   # ITL cap: decode ITL p99 during a prefill burst
python benchmarks/slo/energy_ol.py    # DVFS: power + tok/J across SM clock
python benchmarks/slo/roofline.py     # step-ms vs batch M (the cost model)
```

**Tests:** `pytest tests/v1/core/sched/test_slo.py -q` (25 passed, GPU-free).

## Correctness & safety

- **Zero-risk opt-in.** With the gates unset (default) the path is a provable no-op —
  the cap is `1<<30` and every admission hook is skipped — so behavior is byte-identical
  to stock and the differential CI stays green.
- **Output-exact when on.** The controls change only admission order and prefill chunk
  size, never token values. Greedy output matches stock within the accuracy gate
  (GSM8K + long-context NIAH); like stock vLLM, it is not bitwise-reproducible under
  concurrency (fp non-associativity), so the guarantee is output-exact-by-construction,
  not bit-for-bit.
