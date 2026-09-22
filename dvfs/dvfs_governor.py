#!/usr/bin/env python3
"""SLO-aware DVFS governor for vLLM decode (energy-as-code).

GLM-5.2 decode is memory-latency-bound (warps stalled ~88% on HBM), so at
offered loads below saturation the SM clock has large slack: it can be lowered
to cut GPU power ~25% with ZERO throughput loss while inter-token latency (ITL)
stays under SLO. At the saturated throughput peak, throughput IS clock-sensitive,
so the clock must ride at max there.

Closed-loop control on the windowed MEAN ITL read from vLLM's own
inter_token_latency counters (sum/count deltas -> exact mean, no bucket noise):
  mean ITL > SLO, or running-batch surge   -> jump clock to MAX (protect SLO+throughput)
  mean ITL > setpoint                       -> step clock UP
  mean ITL < setpoint*down_frac             -> step clock DOWN (harvest power), with hysteresis
  else                                      -> hold (deadband)

Output-exact (clock never changes arithmetic). Strictly dominates any static
clock: harvests low-load power like a static-low knob, yet rides to max under
load like static-max -- neither wastes power at low load nor violates SLO/
throughput under a burst.
"""
import time, argparse, subprocess, urllib.request, sys

HOST, PORT = "127.0.0.1", 8000
CLK_STEPS = [700, 800, 900, 1000, 1100, 1200, 1350, 1500, 1650, 1800, 1965]

def fetch():
    with urllib.request.urlopen(f"http://{HOST}:{PORT}/metrics", timeout=5) as r:
        return r.read().decode()

def parse(text):
    s = c = run = 0.0
    for ln in text.splitlines():
        if ln.startswith("vllm:inter_token_latency_seconds_sum"):
            s = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:inter_token_latency_seconds_count"):
            c = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:num_requests_running"):
            run = float(ln.rsplit(" ", 1)[1])
    return s, c, run

def set_clock(mhz, dry):
    if not dry:
        subprocess.run(["sudo", "nvidia-smi", "-lgc", str(mhz)],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def reset_clock(dry):
    if not dry:
        subprocess.run(["sudo", "nvidia-smi", "-rgc"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slo-ms", type=float, default=40.0, help="hard ITL ceiling -> jump to max")
    ap.add_argument("--setpoint-ms", type=float, default=16.0, help="target mean ITL to ride at")
    ap.add_argument("--down-frac", type=float, default=0.85)
    ap.add_argument("--hi-batch", type=int, default=40,
                    help="running batch >= this -> compute-bound regime -> ride to max clock")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--down-holds", type=int, default=2)
    ap.add_argument("--min-clk", type=int, default=1000)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--log", type=str, default="")
    args = ap.parse_args()

    steps = [c for c in CLK_STEPS if c >= args.min_clk]
    idx = len(steps) - 1
    set_clock(steps[idx], args.dry)
    logf = open(args.log, "w") if args.log else sys.stdout
    ps = pc = None
    prev_run = 0.0
    low = 0
    try:
        while True:
            time.sleep(args.interval)
            try:
                s, c, run = parse(fetch())
            except Exception:
                continue
            mean = None
            if ps is not None and c > pc:
                mean = (s - ps) / (c - pc) * 1000.0
            ps, pc = s, c
            surge = run > prev_run + 8
            prev_run = run
            old = idx
            if run >= args.hi_batch or (mean is not None and mean > args.slo_ms) or surge:
                idx = len(steps) - 1; low = 0       # compute-bound / SLO breach / spike -> max
            elif mean is None:                     # no decode traffic -> idle, ease down
                low += 1
                if low >= args.down_holds and idx > 0: idx -= 1; low = 0
            elif mean > args.setpoint_ms:          # above target -> step up
                idx = min(len(steps) - 1, idx + 1); low = 0
            elif mean < args.setpoint_ms * args.down_frac:  # slack -> step down (hysteresis)
                low += 1
                if low >= args.down_holds and idx > 0: idx -= 1; low = 0
            else:
                low = 0
            if idx != old:
                set_clock(steps[idx], args.dry)
            ts = time.strftime("%H:%M:%S")
            mstr = f"{mean:5.1f}" if mean is not None else " idle"
            print(f"{ts} meanITL={mstr}ms run={run:3.0f} clk={steps[idx]:4d}"
                  f"{' *' if idx != old else ''}", file=logf, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        reset_clock(args.dry)

if __name__ == "__main__":
    main()
