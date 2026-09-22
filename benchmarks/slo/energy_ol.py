#!/usr/bin/env python3
"""Open-loop energy gate: fire requests at a fixed rate (below saturation) and
sweep SM clock. Below saturation the served throughput is pinned at the offered
rate regardless of clock (GPU has slack), so downclocking is THROUGHPUT-NEUTRAL:
power drops, tok/J rises, until clock is so low that ITL SLO breaks or the
server can no longer keep up.
"""
import asyncio, json, time, argparse, random, subprocess, urllib.request

HOST, PORT = "127.0.0.1", 8000
MODEL = "glm-5.2-nvfp4"

def gen_counter():
    with urllib.request.urlopen(f"http://{HOST}:{PORT}/metrics", timeout=5) as r:
        for ln in r.read().decode().splitlines():
            if ln.startswith("vllm:generation_tokens_total"):
                return float(ln.rsplit(" ", 1)[1])
    return 0.0

def running_now():
    with urllib.request.urlopen(f"http://{HOST}:{PORT}/metrics", timeout=5) as r:
        for ln in r.read().decode().splitlines():
            if ln.startswith("vllm:num_requests_running"):
                return float(ln.rsplit(" ", 1)[1])
    return 0.0

def total_power():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        text=True)
    return sum(float(x) for x in out.split())

def set_clock(mhz):
    cmd = ["sudo", "nvidia-smi", "-rgc"] if mhz == 0 else ["sudo", "nvidia-smi", "-lgc", str(mhz)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def make_prompt(nints, seed):
    rnd = random.Random(seed)
    return " ".join(str(rnd.randint(0, 1_000_000)) for _ in range(nints))

ITLS = []   # per-token inter-token latencies (ms), collected globally

async def one(prompt, out, salt, k):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": out,
                       "temperature": 0.0, "stream": True, "ignore_eos": True}).encode()
    req = (f"POST /v1/completions HTTP/1.1\r\nHost: {HOST}\r\n"
           f"Content-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    try:
        r, w = await asyncio.open_connection(HOST, PORT)
        w.write(req); await w.drain()
        while True:
            line = await r.readline()
            if line in (b"\r\n", b""): break
        buf = b""; last = None
        while True:
            c = await r.read(16384)
            if not c: break
            buf += c
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1); ln = ln.strip()
                if not ln.startswith(b"data:"): continue
                p = ln[5:].strip()
                if p == b"[DONE]": continue
                try: o = json.loads(p)
                except Exception: continue
                if o.get("choices", [{}])[0].get("text", ""):
                    now = time.perf_counter()
                    if last is not None:
                        ITLS.append((now - last) * 1000)
                    last = now
        w.close()
        try: await w.wait_closed()
        except Exception: pass
    except Exception:
        pass

async def measure_power(secs):
    ps = []; t_end = time.perf_counter() + secs
    while time.perf_counter() < t_end:
        ps.append(total_power()); await asyncio.sleep(0.25)
    return sum(ps)/len(ps)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=6.0)   # req/s open loop
    ap.add_argument("--ints", type=int, default=110)
    ap.add_argument("--out", type=int, default=512)
    ap.add_argument("--clocks", type=str, default="0,1600,1400,1200,1000,800")
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--salt", type=int, default=0)
    args = ap.parse_args()
    clocks = [int(x) for x in args.clocks.split(",")]

    tasks = set(); ctr = [0]; stop = [False]
    async def launcher():
        interval = 1.0 / args.rate
        next = time.perf_counter()
        while not stop[0]:
            ctr[0] += 1; k = ctr[0]
            t = asyncio.create_task(one(make_prompt(args.ints, args.salt*10**7 + k),
                                        args.out, args.salt, k))
            tasks.add(t); t.add_done_callback(tasks.discard)
            next += interval
            dt = next - time.perf_counter()
            if dt > 0: await asyncio.sleep(dt)
    lt = asyncio.create_task(launcher())
    await asyncio.sleep(args.settle + 10)
    print(f"# open-loop rate={args.rate}/s out={args.out} => offered gen ~= {args.rate*args.out:.0f} tok/s")
    print(f"{'clk':>6} {'tok/s':>8} {'power_W':>8} {'tok/J':>8} {'run':>5} {'ITLp50':>7} {'ITLp99':>7}")
    rows = []
    for clk in clocks:
        set_clock(clk)
        await asyncio.sleep(args.settle)
        ITLS.clear()
        g0 = gen_counter(); t0 = time.perf_counter()
        pw = await measure_power(args.window)
        g1 = gen_counter(); t1 = time.perf_counter()
        run = running_now()
        toks = (g1 - g0)/(t1 - t0)
        its = sorted(ITLS)
        p50 = its[len(its)//2] if its else 0
        p99 = its[int(len(its)*0.99)] if its else 0
        rows.append((clk, toks, pw, toks/pw, run, p50, p99))
        print(f"{clk:>6} {toks:>8.1f} {pw:>8.1f} {toks/pw:>8.4f} {run:>5.0f} {p50:>7.2f} {p99:>7.2f}")
    stop[0] = True; lt.cancel()
    for t in list(tasks): t.cancel()
    set_clock(0)
    base = rows[0]
    print("\n# clk  d_tok%  d_pow%  d_tokperJ%")
    for clk, toks, pw, tj, run, p50, p99 in rows:
        print(f"{clk:>6} {100*(toks/base[1]-1):+6.1f} {100*(pw/base[2]-1):+6.1f} {100*(tj/base[3]-1):+8.1f}")

if __name__ == "__main__":
    asyncio.run(main())
