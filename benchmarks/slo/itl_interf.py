import asyncio,json,time,sys,os,statistics as st
HOST,PORT="127.0.0.1",8000
TAG=sys.argv[1]
DEC=int(os.environ.get("DEC","16"))       # steady decode clients
BURST=int(os.environ.get("BURST","24"))   # concurrent long-prefill injections
PWORDS=int(os.environ.get("PWORDS","5000"))
ITL=[]  # (itl_ms, wallclock)
async def decode_client(cid,stop,t0):
    prompt="hello world "*20
    while not stop[0]:
        body=json.dumps({"model":"glm-5.2-nvfp4","prompt":prompt,"max_tokens":256,"temperature":0.0,"ignore_eos":True,"stream":True}).encode()
        req=(f"POST /v1/completions HTTP/1.1\r\nHost:{HOST}\r\nContent-Type:application/json\r\nX-Request-Id:cmpl-int-{cid}-{int(time.time()*1000)%100000}\r\nContent-Length:{len(body)}\r\nConnection:close\r\n\r\n").encode()+body
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
                        if last is not None: ITL.append(((now-last)*1000, now-t0))
                        last=now
            w.close()
        except: await asyncio.sleep(0.1)
async def long_prefill(t0,bid):
    prompt=f"doc{bid} unique "+("data information context "*PWORDS)
    body=json.dumps({"model":"glm-5.2-nvfp4","prompt":prompt,"max_tokens":8,"temperature":0.0,"ignore_eos":True}).encode()
    req=(f"POST /v1/completions HTTP/1.1\r\nHost:{HOST}\r\nContent-Type:application/json\r\nX-Request-Id:cmpl-bulk-{bid}\r\nContent-Length:{len(body)}\r\nConnection:close\r\n\r\n").encode()+body
    try:
        r,w=await asyncio.open_connection(HOST,PORT); w.write(req); await w.drain()
        while True:
            c=await r.read(16384)
            if not c: break
        w.close()
    except: pass
async def main():
    t0=time.perf_counter(); stop=[False]
    decs=[asyncio.create_task(decode_client(i,stop,t0)) for i in range(DEC)]
    await asyncio.sleep(12)  # warmup steady decode
    # measure BEFORE
    base_start=time.perf_counter()-t0
    await asyncio.sleep(6)
    inj_start=time.perf_counter()-t0
    # inject burst of long prefills
    burst=[asyncio.create_task(long_prefill(t0,_)) for _ in range(BURST)]
    await asyncio.gather(*burst)
    inj_end=time.perf_counter()-t0
    await asyncio.sleep(2)
    stop[0]=True
    for d in decs: d.cancel()
    def q(a,x): s=sorted(a); return s[min(len(s)-1,int(x*len(s)))] if s else 0
    base=[i for i,wc in ITL if base_start<=wc<inj_start]
    during=[i for i,wc in ITL if inj_start<=wc<=inj_end]
    print(f"[{TAG}] decode ITL(ms) BEFORE-burst: p50={q(base,.5):.1f} p99={q(base,.99):.1f} (n={len(base)}) | "
          f"DURING {BURST}x{PWORDS}w-prefill-burst: p50={q(during,.5):.1f} p99={q(during,.99):.1f} max={max(during) if during else 0:.0f} (n={len(during)})")
asyncio.run(main())
