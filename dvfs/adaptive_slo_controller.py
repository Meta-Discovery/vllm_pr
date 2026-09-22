#!/usr/bin/env python3
"""SLO-adaptive controller — ONE operator knob: --tail-slo-ms.

Composition (honest: not a new mechanism) of two banked wins, driven by a single
tail-ITL SLO and SELF-CALIBRATING everything else from the live decode roofline:
  - chunk-cap (slo_cfg itl_ms = effective SLO): removes prefill-collision ITL spikes
    so the tail ≈ decode floor; its online-calibrated cost model auto-adapts the
    prefill budget to the current clock.
  - DVFS governor: downclocks to the lowest clock whose floor stays under the SLO.
Closed-loop feasibility BOTH ways:
  - infeasible (pinned at max clock, tail still > effective SLO)  -> RELAX eff SLO
    up to the measured floor (graceful degradation instead of starvation);
  - load eases (sustained slack)                                 -> RE-TIGHTEN eff
    SLO back toward the requested SLO.
The operator sets only the SLO. Relax target, re-tighten trigger/step, the tail
ratio, and the floor estimate are derived from measurements or fixed as sensible
internal constants — NOT knobs. Output-exact (clock + chunk-timing only).
"""
import os, time, argparse, subprocess, urllib.request, sys

HOST, PORT = "127.0.0.1", 8000
CLK_STEPS = [1000, 1100, 1200, 1350, 1500, 1650, 1800, 1965]

# --- internal constants (self-tuning policy; NOT operator knobs) ---
INTERVAL_S      = 1.0     # control period
SETPOINT_FRAC   = 0.55    # downclock until mean ITL ~ this fraction of the (effective) SLO
DOWN_HOLDS      = 2       # hysteresis before stepping the clock down
MIN_PREFILL     = 256     # chunk-cap floor (tokens/step)
INFEAS_HOLDS    = 5       # sustained intervals pinned-at-max-and-missing before RELAX
RELAX_MARGIN    = 1.10    # relax target = measured floor x this
RETIGHTEN_HOLDS = 8       # sustained slack intervals before a RE-TIGHTEN step
RETIGHTEN_FRAC  = 0.65    # slack = tail below this fraction of the effective SLO (headroom)
RETIGHTEN_FRAC_STEP = 0.15  # re-tighten step = this fraction of the effective SLO (self-scaled)
TAIL_RATIO_INIT = 1.5     # p99/mean seed; self-calibrated online from the metrics
CFG_PATH    = os.environ.get("SLO_CFG_PATH", "/tmp/slo_cfg")
STATUS_PATH = os.environ.get("SLO_STATUS_PATH", "/tmp/slo_status")


def fetch():
    with urllib.request.urlopen(f"http://{HOST}:{PORT}/metrics", timeout=5) as r:
        return r.read().decode()


def parse(text):
    s = c = run = 0.0
    buckets = {}
    for ln in text.splitlines():
        if ln.startswith("vllm:inter_token_latency_seconds_sum"):
            s = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:inter_token_latency_seconds_count"):
            c = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:inter_token_latency_seconds_bucket"):
            try:
                le = ln.split('le="', 1)[1].split('"', 1)[0]
                buckets[float("inf") if le == "+Inf" else float(le)] = float(ln.rsplit(" ", 1)[1])
            except Exception:
                pass
        elif ln.startswith("vllm:num_requests_running"):
            run = float(ln.rsplit(" ", 1)[1])
    return s, c, run, buckets


def windowed_p99(prev_b, cur_b, q=0.99):
    if not prev_b or not cur_b:
        return None
    les = sorted(k for k in cur_b if k != float("inf")) + [float("inf")]
    deltas = [(le, max(0.0, cur_b.get(le, 0.0) - prev_b.get(le, 0.0))) for le in les]
    total = sum(d for _, d in deltas)
    if total < 5:
        return None
    tgt = q * total; cum = 0.0
    for i, (le, d) in enumerate(deltas):
        cum += d
        if cum >= tgt:
            return (le * 1000.0) if le != float("inf") else (les[i - 1] * 1000.0)
    return None


def set_clock(mhz, dry):
    if not dry:
        subprocess.run(["sudo", "nvidia-smi", "-lgc", str(mhz)],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail-slo-ms", type=float, required=True,
                    help="THE knob: operator's p99 inter-token-latency SLO (ms). Everything "
                         "else self-calibrates.")
    # plumbing (not tuning): where to write, and a test/no-op switch
    ap.add_argument("--log", type=str, default="")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    T = args.tail_slo_ms
    eff_T = T
    steps = CLK_STEPS
    idx = len(steps) - 1
    set_clock(steps[idx], args.dry)
    logf = open(args.log, "w") if args.log else sys.stdout

    def write_cfg(itl):
        try:
            with open(CFG_PATH, "w") as f:
                f.write(f"{itl:.0f},{MIN_PREFILL},{MIN_PREFILL},1000000,0,0,0")
        except Exception:
            pass

    def write_status(feasible, floor, reason, eff):
        try:
            with open(STATUS_PATH, "w") as f:
                f.write(f'{{"slo_ms": {T:.0f}, "feasible": {str(feasible).lower()}, '
                        f'"floor_est_ms": {floor:.1f}, "effective_slo_ms": {eff:.0f}, '
                        f'"reason": "{reason}"}}')
        except Exception:
            pass

    write_cfg(eff_T)
    write_status(True, 0.0, "init", eff_T)

    prev_s = prev_c = prev_b = None
    low = infeas = slack = 0
    ema_mean = None
    tail_ratio = TAIL_RATIO_INIT     # self-calibrated online below
    try:
        while True:
            time.sleep(INTERVAL_S)
            try:
                s, c, run, buckets = parse(fetch())
            except Exception:
                continue
            mean = (s - prev_s) / (c - prev_c) * 1000.0 if (prev_c is not None and c > prev_c) else None
            p99 = windowed_p99(prev_b, buckets)
            prev_s, prev_c, prev_b = s, c, buckets
            if mean is None:
                continue

            # self-calibrate the p99/mean tail ratio from live metrics (bounded)
            if p99 is not None and mean > 1e-6:
                r = min(2.2, max(1.15, p99 / mean))
                tail_ratio = 0.9 * tail_ratio + 0.1 * r
            ema_mean = mean if ema_mean is None else (0.7 * ema_mean + 0.3 * mean)
            tail = ema_mean * tail_ratio          # estimated p99 ITL (self-derived ratio)

            setpoint = eff_T * SETPOINT_FRAC      # governor targets the EFFECTIVE SLO
            old = idx
            if mean > eff_T * 0.9:
                idx = len(steps) - 1; low = 0
            elif mean > setpoint:
                idx = min(idx + 1, len(steps) - 1); low = 0
            elif mean < setpoint * 0.85:
                low += 1
                if low >= DOWN_HOLDS and idx > 0:
                    idx -= 1; low = 0
            else:
                low = 0
            if idx != old:
                set_clock(steps[idx], args.dry)
            pinned_max = (idx == len(steps) - 1)

            # closed-loop feasibility, both directions
            if pinned_max and tail > eff_T:                       # can't meet even eff_T -> RELAX up
                infeas += 1; slack = 0
                if infeas >= INFEAS_HOLDS:
                    new_eff = max(T, tail * RELAX_MARGIN)         # relax to the MEASURED floor
                    if new_eff > eff_T + 0.5:
                        eff_T = new_eff; write_cfg(eff_T)
                        print(f"{time.strftime('%H:%M:%S')} INFEASIBLE req={T:.0f} floor~{tail:.0f}"
                              f" -> RELAX eff={eff_T:.0f}", file=logf, flush=True)
                    feasible = (eff_T <= T + 0.5)
                    write_status(feasible, tail,
                                 f"requested {T:.0f}ms infeasible (floor ~{tail:.0f}); operating at "
                                 f"{eff_T:.0f}ms" if not feasible else f"meeting {T:.0f}ms at max clock",
                                 eff_T)
                    infeas = 0
            elif eff_T > T + 0.5 and (not pinned_max) and tail < eff_T * RETIGHTEN_FRAC:
                slack += 1; infeas = 0                            # sustained slack -> RE-TIGHTEN down
                if slack >= RETIGHTEN_HOLDS:
                    eff_T = max(T, eff_T - RETIGHTEN_FRAC_STEP * eff_T)
                    write_cfg(eff_T)
                    feasible = (eff_T <= T + 0.5)
                    write_status(feasible, tail,
                                 f"load eased -> re-tightened to {eff_T:.0f}ms"
                                 + ("" if feasible else f" (still > requested {T:.0f})"), eff_T)
                    print(f"{time.strftime('%H:%M:%S')} load eased -> RE-TIGHTEN eff={eff_T:.0f}"
                          f" (toward {T:.0f})", file=logf, flush=True)
                    slack = 0
            else:
                infeas = max(0, infeas - 1); slack = max(0, slack - 1)

            pstr = f"{p99:5.1f}" if p99 is not None else "  n/a"
            print(f"{time.strftime('%H:%M:%S')} req={T:.0f} eff={eff_T:.0f} mean={mean:5.1f} p99={pstr} "
                  f"tail~{tail:.0f} r={tail_ratio:.2f} run={run:.0f} clk={steps[idx]}"
                  f"{' *' if idx != old else ''}", file=logf, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.dry:
            subprocess.run(["sudo", "nvidia-smi", "-rgc"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
