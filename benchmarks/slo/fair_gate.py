#!/usr/bin/env python3
"""Fairness gate: does a heavy batch client (A) starve a light interactive client (B)
under vLLM FCFS? Pure-stdlib asyncio streaming client (no deps, single-thread -> no GIL
contention). Measures B's TTFT/E2E while A floods, vs B alone.
"""
import asyncio, json, time, argparse, random, sys

HOST, PORT = "127.0.0.1", 8000
MODEL = "glm-5.2-nvfp4"
REQ_TIMEOUT = 60.0

def make_prompt(ntok, seed):
    # unique-ish random word prompt to avoid prefix cache (approx ntok tokens ~ ntok words)
    rnd = random.Random(seed)
    return " ".join(str(rnd.randint(0, 1_000_000)) for _ in range(ntok))

_REQN = {"A": 0, "B": 0}
async def one_request(prompt, max_tokens, tag, results):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "ignore_eos": True,
        "stream_options": {"include_usage": False},
    }).encode()
    _REQN[tag] += 1
    xrid = f"client{tag}-{_REQN[tag]}"   # VTC groups by client key 'client{A,B}'
    req = (f"POST /v1/completions HTTP/1.1\r\nHost: {HOST}\r\n"
           f"Content-Type: application/json\r\nX-Request-Id: {xrid}\r\n"
           f"Content-Length: {len(body)}\r\n"
           f"Connection: close\r\n\r\n").encode() + body
    t0 = time.perf_counter()
    state = {"ttft": None, "ntok": 0}
    async def _run():
        reader, writer = await asyncio.open_connection(HOST, PORT)
        writer.write(req); await writer.drain()
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"") : break
        buf = b""
        while True:
            chunk = await reader.read(4096)
            if not chunk: break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"): continue
                payload = line[5:].strip()
                if payload == b"[DONE]": continue
                try: obj = json.loads(payload)
                except Exception: continue
                txt = obj.get("choices", [{}])[0].get("text", "")
                if txt:
                    if state["ttft"] is None: state["ttft"] = time.perf_counter() - t0
                    state["ntok"] += 1
        writer.close()
        try: await writer.wait_closed()
        except Exception: pass
    try:
        await asyncio.wait_for(_run(), timeout=REQ_TIMEOUT)
    except asyncio.TimeoutError:
        results.append({"tag": tag, "ttft": state["ttft"], "e2e": REQ_TIMEOUT,
                        "ntok": state["ntok"], "t0": t0, "timeout": True})
        return
    except Exception as e:
        results.append({"tag": tag, "err": str(e)[:80], "t0": t0})
        return
    e2e = time.perf_counter() - t0
    results.append({"tag": tag, "ttft": state["ttft"], "e2e": e2e, "ntok": state["ntok"], "t0": t0})

async def heavy_client(conc, dur, max_tokens, start_at, results, aprompt, salt):
    # Continuous backlogged load: exactly `conc` requests in flight for `dur` seconds.
    await asyncio.sleep(start_at)
    t_end = time.perf_counter() + dur
    ctr = [0]
    async def worker():
        while time.perf_counter() < t_end:
            ctr[0] += 1; k = ctr[0]
            await one_request(make_prompt(aprompt, salt*7_000_000 + k), max_tokens, "A", results)
    await asyncio.gather(*[worker() for _ in range(conc)], return_exceptions=True)

async def light_client(rate, dur, max_tokens, start_at, results, salt):
    # steady interactive: one short req every 1/rate sec, over the measurement window
    await asyncio.sleep(start_at)
    i = 0; tasks = []
    t_end = time.perf_counter() + dur
    while time.perf_counter() < t_end:
        i += 1
        tasks.append(asyncio.create_task(
            one_request(make_prompt(128, salt*3_000_000 + i), max_tokens, "B", results)))
        await asyncio.sleep(1.0 / rate)
    await asyncio.gather(*tasks, return_exceptions=True)
    stop_evt.set()

def pct(xs, p):
    xs = sorted(xs)
    if not xs: return None
    k = (len(xs)-1)*p/100
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["alone", "contend"], required=True)
    ap.add_argument("--dur", type=float, default=30)
    ap.add_argument("--brate", type=float, default=2.0)   # B interactive rate
    ap.add_argument("--bmax", type=int, default=128)
    ap.add_argument("--aburst", type=int, default=256)    # A total requests
    ap.add_argument("--aconc", type=int, default=64)      # A concurrency (fills slots)
    ap.add_argument("--amax", type=int, default=512)
    ap.add_argument("--astart", type=float, default=3.0)  # A starts after warmup
    ap.add_argument("--aprompt", type=int, default=400)   # A prompt size (words)
    ap.add_argument("--bstart", type=float, default=5.0)  # B starts (after A saturates)
    ap.add_argument("--salt", type=int, default=0)        # per-arm prompt salt (distinct prompts)
    args = ap.parse_args()
    results = []; stop = asyncio.Event()
    # A runs long enough to cover B's whole measurement window (+tail).
    clients = [light_client(args.brate, args.dur, args.bmax, args.bstart, results, args.salt)]
    if args.mode == "contend":
        a_dur = args.bstart + args.dur + 5 - args.astart
        clients.append(heavy_client(args.aconc, a_dur, args.amax, args.astart, results, args.aprompt, args.salt))
    t0 = time.perf_counter()
    await asyncio.gather(*clients, return_exceptions=True)
    wall = time.perf_counter() - t0
    # For starvation, count a request that never produced a token as fully starved:
    # its effective TTFT = its total wait (>= REQ_TIMEOUT if it timed out).
    def eff_ttft(r):
        if r.get("ttft") is not None: return r["ttft"]*1000
        return REQ_TIMEOUT*1000  # never got a token in the window
    Ball = [r for r in results if r["tag"]=="B" and "err" not in r]
    A = [r for r in results if r["tag"]=="A" and "err" not in r and r.get("ttft") is not None]
    Berr = [r for r in results if r["tag"]=="B" and "err" in r]
    Bto = [r for r in Ball if r.get("timeout")]
    def summ(rs, use_eff=False):
        if not rs: return "none"
        ttfts = [eff_ttft(r) if use_eff else r["ttft"]*1000 for r in rs if use_eff or r.get("ttft") is not None]
        e2es = [r["e2e"]*1000 for r in rs]
        toks = sum(r["ntok"] for r in rs)
        return (f"n={len(rs)} TTFT p50={pct(ttfts,50):.0f} p90={pct(ttfts,90):.0f} p99={pct(ttfts,99):.0f}ms "
                f"E2E p50={pct(e2es,50):.0f} p99={pct(e2es,99):.0f}ms tok={toks}")
    print(f"MODE={args.mode} wall={wall:.1f}s")
    print(f"  B (light): {summ(Ball, use_eff=True)}  errs={len(Berr)} timeouts={len(Bto)}")
    if A: print(f"  A (heavy): {summ(A)} thr={sum(r['ntok'] for r in A)/wall:.0f} tok/s")

asyncio.run(main())
