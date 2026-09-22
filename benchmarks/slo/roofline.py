#!/usr/bin/env python3
"""Decode-cost roofline: median pure-decode ITL vs batch M. Fits step = base + k*M^p."""
import asyncio,json,time,sys
HOST,PORT="127.0.0.1",8000
async def stream(stop,itls):
    prompt="hello world "*10
    while not stop[0]:
        body=json.dumps({"model":"glm-5.2-nvfp4","prompt":prompt,"max_tokens":400,"temperature":0.0,"ignore_eos":True,"stream":True}).encode()
        req=(f"POST /v1/completions HTTP/1.1\r\nHost:{HOST}\r\nContent-Type:application/json\r\nContent-Length:{len(body)}\r\nConnection:close\r\n\r\n").encode()+body
        try:
            r,w=await asyncio.open_connection(HOST,PORT); w.write(req); await w.drain()
            while True:
                l=await r.readline()
                if l in (b"\r\n",b""): break
            buf=b""; last=None
            while True:
                c=await r.read(8192)
                if not c: break
                buf+=c
                while b"\n" in buf:
                    ln,buf=buf.split(b"\n",1); ln=ln.strip()
                    if not ln.startswith(b"data:"): continue
                    p=ln[5:].strip()
                    if p==b"[DONE]": continue
                    try: o=json.loads(p)
                    except: continue
                    if o.get("choices",[{}])[0].get("text",""):
                        now=time.perf_counter()
                        if last is not None: itls.append((now-last)*1000)
                        last=now
            w.close()
        except: await asyncio.sleep(0.05)
async def measure(M):
    stop=[False]; itls=[]
    ts=[asyncio.create_task(stream(stop,itls)) for _ in range(M)]
    await asyncio.sleep(8)  # ramp
    itls.clear(); await asyncio.sleep(10)
    stop[0]=True
    for t in ts: t.cancel()
    await asyncio.sleep(0.3)
    itls.sort()
    return itls[len(itls)//2] if itls else 0
async def main():
    print(f"{'M':>4} {'medITL_ms':>10}")
    rows=[]
    for M in [1,4,8,16,32,48,64]:
        med=await measure(M); rows.append((M,med)); print(f"{M:>4} {med:>10.2f}",flush=True)
    # fit base + k*sqrt(M) and base + k*M (least squares, simple)
    import math
    def fit(xs,ys,f):
        # linear regression ys = a + b*f(x)
        n=len(xs); fx=[f(x) for x in xs]
        mx=sum(fx)/n; my=sum(ys)/n
        b=sum((fx[i]-mx)*(ys[i]-my) for i in range(n))/sum((fx[i]-mx)**2 for i in range(n))
        a=my-b*mx
        ss=sum((ys[i]-(a+b*fx[i]))**2 for i in range(n)); tot=sum((y-my)**2 for y in ys)
        return a,b,1-ss/tot
    xs=[r[0] for r in rows]; ys=[r[1] for r in rows]
    a1,b1,r1=fit(xs,ys,lambda x:x)
    a2,b2,r2=fit(xs,ys,lambda x:math.sqrt(x))
    print(f"\nLINEAR  step={a1:.2f}+{b1:.3f}*M     R2={r1:.4f}")
    print(f"SQRT-M  step={a2:.2f}+{b2:.3f}*sqrt(M) R2={r2:.4f}")
asyncio.run(main())
