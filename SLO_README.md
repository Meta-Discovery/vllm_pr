# SLO-aware, output-exact serving controls for vLLM

Base: vLLM `bca7bea2405127bd5291bb6fffa679bdcd8f6dd9`. One operator knob — a tail
inter-token-latency (ITL) SLO — that spends idle hardware slack on lower latency,
fairer sharing, and lower power. Output-exact; a no-op when off.

## Why it works

Decode on GLM-5.2-NVFP4 (8×B200) is memory-latency-bound: `step_ms ≈ 5.9 + 0.49·√M`
(R²=0.99). So the **prefill chunk size** co-scheduled with decodes and the **SM clock**
are slack you can spend to hit a latency target without changing the math.

## The changes (in order)

| # | Change (file · gate) | Design | Helps |
|---|---|---|---|
| 1 | **ITL chunk-cap** · `scheduler.py` · `VLLM_SLO_ENABLE=1` | Each step, cap prefill tokens co-scheduled with decodes so predicted step time ≤ SLO. (Stock's `long_prefill_token_threshold` is per-request — can't bound per-*step* time under K prefills.) | decode p99 tail |
| 2 | **VTC fair queuing** · `scheduler.py` + `slo.py` · `VLLM_SLO_VTC=1` | Weighted per-client virtual-time admission; under-served clients admit first. (Stock `priority` is static per-request.) | light-client first-token |
| 3 | **DVFS governor** · `dvfs/` · host tool | Lower the SM clock during decode slack, raise it during prefill bursts. | power |

All gated; unset ⇒ byte-identical stock. ~500 lines, no kernel changes.

## Performance (vs **tuned** stock vLLM, output-exact)

**System** — 85-min true eval, rising arrival-rate sweep:
p99 decode **2.94×** · first-token **3.61×** · power **−21%** · throughput **1.01×** · geomean **1.92×**.

**Component** — each mechanism alone, fixed rates:

| Measure | Tuned stock | This PR | Change |
|---|---|---|---|
| p99 decode latency, real workload | 141–337 ms | 22–26 ms | **6–13×** |
| Light-client first-token under heavy | 9654 ms | 1114 ms | **8.7×** |
| p99 under 8 concurrent long prefills | 221.6–507.7 ms | 24.3 ms | **9–21×** |
| Power at equal throughput | — | — | **−34%** |

## Run

```bash
# OFF (baseline)
vllm serve nvidia/GLM-5.2-NVFP4 --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8_e4m3 --scheduling-policy priority --max-num-seqs 64 --port 8000
# ON (same command, gated prefix)
VLLM_SLO_ENABLE=1 VLLM_SLO_VTC=1 VLLM_SLO_DECODE_EXP=0.5 vllm serve ...
# DVFS governor (host-side, needs NVML)
python dvfs/adaptive_slo_controller.py --tail-slo-ms 40
# Tests
pytest tests/v1/core/sched/test_slo.py -q          # 25, GPU-free
# Repro harnesses
python benchmarks/slo/{fair_gate,itl_interf,energy_ol,roofline}.py
```

## Files

`vllm/v1/core/sched/scheduler.py` (cap + VTC, +448) · `slo.py` (cost model + VTC) ·
`tests/v1/core/sched/test_slo.py` · `dvfs/` (governor) · `benchmarks/slo/` (repro).

## Status

Research build for discussion, not merge-ready. Component numbers reproduce from this
repo; system numbers are from the 85-min eval (counters from nsys). Before upstream:
tests exercise `slo.py` not the inlined scheduler; a `/tmp/slo_cfg` live-config hook
should be removed; the DVFS tool needs a SIGTERM clock-restore and a `windowed_p99`
de-cumulation fix; the cap is best-effort when the decode batch alone exceeds the SLO.
