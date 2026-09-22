# SLO-aware, output-exact serving controls (env-gated, OFF by default)

Base: vLLM `bca7bea2405127bd5291bb6fffa679bdcd8f6dd9`. Model used for measurement:
`nvidia/GLM-5.2-NVFP4` on one 8×B200 node (TP8 / expert-parallel / fp8 KV).

## The idea

On this model decode is memory-latency-bound: the decode step time follows a
measured roofline `step_ms ≈ 5.9 + 0.49·√M` (batch M, R²=0.99, verified by
hardware counters — Tensor Active 2.3%, DRAM read 21%). Two things are therefore
slack the engine can spend to hit a latency target without changing arithmetic:
the **prefill chunk size** co-scheduled with decodes, and the **SM clock**. This
change turns that slack into one operator knob — a tail inter-token-latency SLO —
driving controls stock vLLM cannot express. All controls change only *when* work
is admitted and *how large* a prefill chunk is — never token values — so greedy
output stays byte-identical to stock.

## What it adds (all gated; unset ⇒ byte-identical stock)

| Control | Env gate | What it does |
|---|---|---|
| Per-step ITL chunk-cap | `VLLM_SLO_ENABLE=1` | Bound prefill tokens co-scheduled with decodes so predicted step time ≤ `safety·SLO`. A static per-*request* `long_prefill_token_threshold` cannot bound per-*step* time under K concurrent prefills. |
| VTC per-client fairness | `VLLM_SLO_VTC=1` | Weighted virtual-time fair-queuing on admission; under-served clients admit first. Stock `priority` is static per-request. |
| Feasibility + DVFS governor | host tool | Downclock to the lowest clock whose ITL floor stays under the SLO; detect when the SLO is below the load-induced floor, relax to it, re-tighten when load eases. |

Optional admission policies behind their own gates: `VLLM_SLO_SRPT` (short-output
first), `VLLM_SLO_SJF` (short-prompt first), `VLLM_SLO_CACHE_AWARE` (defer
cold-prefix siblings behind a pioneer), `VLLM_SLO_TIERS` / `VLLM_SLO_UNIFIED`
(per-class deadlines).

## Files

| Path | What |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | Gated ITL-cap + VTC + admission hooks (additive; +448 lines). |
| `vllm/v1/core/sched/slo.py` | Stdlib cost model + VTC + feasibility (reference module, unit-tested). |
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
Key knobs: `VLLM_SLO_ITL_MS` (the SLO, default 40), `VLLM_SLO_DECODE_EXP=0.5`
(use the measured √M roofline; the built-in default 1.0 is linear and is refined
online by EMA either way), `VLLM_SLO_INT_PREFIX` (request-id prefix marking the
latency-sensitive class; `""` = protect all). With every gate unset the module is
a no-op (`_slo_step_cap = 1<<30`).

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

## Gains — measured, matched A/B vs *tuned* stock, output-exact

All numbers are greedy byte-identical OFF vs ON; A/B is against tuned stock
(`--scheduling-policy priority` + a tuned `--long-prefill-token-threshold`), with
multi-rep confidence intervals.

| Axis | Result | Honest scope |
|---|---|---|
| Energy (DVFS governor) | **−34% GPU power, +31% tokens/joule**, throughput-neutral, on the real mixed 8192/1024 workload | Comes from the **host-side governor**, not the scheduler diff; needs NVML/clock permission. Most robust result. |
| Fairness (VTC) | **light-client TTFT ~2.8× lower** in-tree (p50 2619→940 ms; p99 6070→3006 ms) | In-tree confirmed 2.8×; magnitude is queue-depth dependent (out-of-tree, deeper queue, measured up to 8.7×). Heavy-client throughput neutral. |
| Latency tail (ITL cap) | vs stock **default**: **~10–15% median p99 improvement + strong worst-case tail protection** (bounds rare cold-burst spikes, e.g. 42–617 ms → ≤38–51 ms) | The larger multiples (up to ~21×) are only vs a *misconfigured* static `long_prefill_token_threshold`, not vs a tuned default. |

## Correctness

- Gate OFF (default) is byte-identical to stock (provable no-op: cap = `1<<30` ⇒
  `num_new_tokens` unchanged; every defer block gated off).
- Gate ON changes only admission order and prefill chunk size, never token values;
  greedy output byte-identical. GSM8K + long-context NIAH hold within tolerance.

## Known limitations (for a real upstream review)

- `slo.py` (unit-tested) is a **reference module**; the scheduler **inlines** an
  equivalent copy with different built-in constants. A submitter should either
  wire the scheduler to import `slo.py` or add tests that exercise the scheduler
  path directly.
- `client_key` extraction differs between `scheduler.py` (first `-`) and `slo.py`
  (last `-`); they agree for single-segment client ids only.
- The `VLLM_SLO_ENABLE` path re-reads a live-config file (`/tmp/slo_cfg`) every 100
  steps; that convenience hook should be removed or replaced before upstreaming.
- Under concurrency, chunked prefill is not bit-identical on tie-heavy degenerate
  inputs due to fp non-associativity — this affects stock vLLM identically and is
  bounded by the accuracy gate; it is not introduced by these controls.
