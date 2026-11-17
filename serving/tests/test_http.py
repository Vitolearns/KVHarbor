"""Integration checks of queues, deadlines, faults and disconnects over TCP."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def poll(predicate, seconds=5):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        try:
            value=predicate()
            if value:
                return value
        except OSError:
            pass
        time.sleep(.015)
    raise AssertionError("condition did not become true")


def test(binary):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",0))
        port=probe.getsockname()[1]
    base=f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile() as log:
        p=subprocess.Popen([binary,"--test-backend","--port",str(port),"--workers","1","--queue","1",
                            "--stream-buffer-bytes","32768","--write-timeout-ms","500"],stdout=log,stderr=log)
        def get(path):
            with urllib.request.urlopen(base+path,timeout=1) as r:
                return json.load(r)
        def post(body=None, timeout_ms=None):
            body={"stream":True,**(body or {})}
            headers={"Content-Type":"application/json"}
            if timeout_ms is not None:
                headers["X-Request-Timeout-Ms"]=str(timeout_ms)
            req=urllib.request.Request(base+"/v1/chat/completions",data=json.dumps(body).encode(),headers=headers)
            return urllib.request.urlopen(req,timeout=6)
        def status(body=None,timeout_ms=None):
            try:
                with post(body,timeout_ms) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code
        def state(active=None,queued=None):
            s=get("/state")
            return s if (active is None or s["active"]==active) and (queued is None or s["tasks"]["llm"]["queued"]==queued) else None
        try:
            assert poll(lambda:get("/health"))["runtimes"]["llm"]["backend"]=="test-double"
            with post({"max_tokens":2}) as r:
                assert b"[DONE]" in r.read()
            with ThreadPoolExecutor(max_workers=1) as pool:
                with post({"test_delay_ms":80,"max_tokens":64}) as active:
                    assert b"data:" in active.readline()
                    queued=pool.submit(status,{"max_tokens":2},350)
                    poll(lambda:state(active=1,queued=1))
                    assert status()==429
                    assert queued.result(timeout=2)==408
                s=poll(lambda:state(active=0,queued=0))
            assert s["started"]==s["slot_releases"]==2, s
            assert s["outcomes"]["deadline"]==1, s
            assert s["outcomes"]["client_disconnected"]==1, s
            assert status({"stream": "true"}) == 400
            assert status(timeout_ms="invalid") == 400
            for malformed in ("8", None, True, False, 1.9, 2.0, 2**40, 2**64 - 1):
                assert status({"max_tokens": malformed}) == 400
            assert status({"test_throw":True})==502
            with post({"max_tokens":2}) as r:
                assert b"[DONE]" in r.read()

            # Deliberately stop reading after sending a large stream request.
            with socket.socket() as slow:
                slow.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4096)
                slow.connect(("127.0.0.1",port))
                body=json.dumps({"stream":True,"max_tokens":2048,"test_delay_ms":0,"test_chunk_bytes":8192}).encode()
                slow.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()+body)
                poll(lambda:state(active=1))
                s=poll(lambda:state(active=0),seconds=5)
                assert s["peak_buffer_bytes"]<=32768, s
                assert s["outcomes"].get("backpressure",0)+s["outcomes"].get("client_write_failed",0)==1, s

            with post({"test_delay_ms":100,"max_tokens":64}) as active:
                poll(lambda:state(active=1))
                p.terminate()
                assert p.wait(timeout=5)==0
            log.seek(0)
            rows=[]
            for line in log.read().decode(errors="replace").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
            terminal=[r for r in rows if "id" in r]
            shutdown=rows[-1]["shutdown"]
            assert shutdown["active"]==0
            assert len(terminal)==len({r["id"] for r in terminal})==shutdown["accepted"], terminal
            assert sum(r["slot_released"] for r in terminal)==shutdown["started"]==shutdown["slot_releases"], terminal
            print(json.dumps({"scope":"test-double","status":"ok","shutdown":shutdown,"terminal_records":terminal}))
        except BaseException:
            log.seek(0)
            print(log.read().decode(errors="replace"))
            raise
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary",type=Path)
    test(str(parser.parse_args().binary.resolve()))
