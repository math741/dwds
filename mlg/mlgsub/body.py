"""ZERO-ASSUMPTION AWAKENING (manifesto XIX-XXI, LXXIII).

Nobody tells the MLG what it is running on. It asks the driver (REPORTED),
then runs experiments on its own body (MEASURED) and only then states traits
(MEASURED or INFERRED), each linked to the measurements that prove it:

    (Body)-[:HAS_TRAIT]->(Trait)-[:PROVEN_BY]->(Measurement)-[:PART_OF]->(Experiment)

Transitions are not sampled blindly: when bandwidth falls off a cliff between
two sizes, the MLG bisects that interval (XVIII) instead of testing where it
already knows the answer.
"""
import math
import os
import platform
import shutil
import subprocess
import threading
import time

import torch

DTYPES = {"fp32": torch.float32, "tf32": torch.float32,
          "bf16": torch.bfloat16, "fp16": torch.float16}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def per_call(fn, dev, min_time=0.02, repeats=3):
    """Median seconds per call; the inner loop grows until timing is stable."""
    fn()
    _sync(dev)
    k = 1
    while True:
        t = time.perf_counter()
        for _ in range(k):
            fn()
        _sync(dev)
        el = time.perf_counter() - t
        if el >= min_time or k >= 1 << 16:
            break
        k *= 2
    times = [el / k]
    for _ in range(repeats - 1):
        t = time.perf_counter()
        for _ in range(k):
            fn()
        _sync(dev)
        times.append((time.perf_counter() - t) / k)
    times.sort()
    return times[len(times) // 2]


def _smi(field):
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    return float(out.splitlines()[0])


class EnergyMeter:
    """Integrates board power over a block; unavailable without a sensor,
    in which case energy is NOT MEASURED rather than invented."""

    def __init__(self, device):
        self.read = self.temp = None
        if device.type == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
                self.read = lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                self.temp = lambda: float(pynvml.nvmlDeviceGetTemperature(h, 0))
            except Exception:
                if shutil.which("nvidia-smi"):
                    self.read = lambda: _smi("power.draw")
                    self.temp = lambda: _smi("temperature.gpu")
        self.available = self.read is not None

    def __enter__(self):
        self.joules = 0.0
        self.peak_w = 0.0
        self.seconds = 0.0
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        if self.available:
            self._last_t, self._last_w = self._t0, self.read()
            self._th = threading.Thread(target=self._run, daemon=True)
            self._th.start()
        return self

    def _run(self):
        while not self._stop.wait(0.1):
            try:
                now, w = time.perf_counter(), self.read()
            except Exception:
                continue
            self.joules += 0.5 * (self._last_w + w) * (now - self._last_t)
            self.peak_w = max(self.peak_w, w)
            self._last_t, self._last_w = now, w

    def __exit__(self, *exc):
        end = time.perf_counter()
        self._stop.set()
        if self.available:
            self._th.join()
            self.joules += self._last_w * (end - self._last_t)
        self.seconds = end - self._t0
        return False

    @property
    def watts(self):
        return self.joules / self.seconds if self.available and self.seconds else None


class BodyModel:
    """What the MLG believes about one body. Every number here has a node."""

    def __init__(self, id_, device, kind):
        self.id = id_
        self.device = device
        self.kind = kind  # PHYSICAL or SIMULATED
        self.name = ""
        self.usable_bytes = None
        self.dram_gbs = None
        self.dtypes = {}  # name -> {"curve": [(n, tflops)], "err": float, "range_ok": bool}
        self.energy = False
        self.efficiency = 0.35  # self-model: real/peak throughput while training
        self.mem_factor = 1.0  # self-model: real/predicted peak training memory
        self.traits = {}  # trait name -> node id
        self.deploy_bytes = None  # simulated bodies: brain memory budget
        self.latency_ms = None  # simulated bodies: per-byte latency budget
        self.lab = None

    def tflops(self, dtype, n):
        curve = self.dtypes[dtype]["curve"]
        if n <= curve[0][0]:
            return curve[0][1]
        for (n0, t0), (n1, t1) in zip(curve, curve[1:]):
            if n <= n1:
                f = math.log(n / n0) / math.log(n1 / n0)
                return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
        return curve[-1][1]

    def summary(self):
        return {
            "body": self.id, "kind": self.kind, "name": self.name,
            "usable_GiB": round(self.usable_bytes / 2**30, 2) if self.usable_bytes else None,
            "dram_GBps": round(self.dram_gbs, 1) if self.dram_gbs else None,
            "peak_TFLOPS": {k: round(max(t for _, t in v["curve"]), 2) for k, v in self.dtypes.items()},
            "energy_sensor": self.energy,
        }


def _fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def awaken(sub, quick=False, log=print):
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    body = BodyModel(f"body:{dev.type}0", dev, "PHYSICAL")
    counter = [0]

    def experiment(name, hypothesis):
        eid = f"exp:{body.id}:{name}"
        sub.obj("Experiment", eid, origin="body.awaken", epistemic="OBSERVED",
                name=name, hypothesis=hypothesis)
        return eid

    def measure(eid, metric, value, unit, level="MEASURED", **ctx):
        counter[0] += 1
        mid = f"m:{body.id}:{counter[0]:04d}"
        sub.obj("Measurement", mid, origin=eid, epistemic=level,
                metric=metric, value=value, unit=unit, **ctx)
        sub.link(mid, "PART_OF", eid)
        sub.link(mid, "MEASURED_ON", body.id)
        return mid

    def trait(name, value, unit, level, proof, **props):
        tid = f"trait:{body.id}:{name}"
        sub.obj("Trait", tid, origin="body.awaken", epistemic=level,
                name=name, value=value, unit=unit, **props)
        sub.link(body.id, "HAS_TRAIT", tid)
        for mid in proof:
            sub.link(tid, "PROVEN_BY", mid)
        body.traits[name] = tid
        return tid

    # E1. Existence: what the driver claims. Claims are REPORTED, not facts.
    sub.obj("Body", body.id, origin="body.awaken", epistemic="REPORTED",
            kind="PHYSICAL", device=dev.type)
    sub.event("AWAKENING", body.id, note="no architecture given; discovering body")
    log(f"[awake] compute device found: {dev.type}. I will not trust its name; I will measure it.")
    e1 = experiment("existence", "what does the platform claim this body is?")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        body.name = p.name
        claims = {"name": p.name, "reported_memory": p.total_memory,
                  "multiprocessors": p.multi_processor_count,
                  "compute_capability": f"{p.major}.{p.minor}"}
    else:
        body.name = platform.processor() or platform.machine()
        claims = {"name": body.name, "logical_cpus": os.cpu_count()}
    for k, v in claims.items():
        trait(k, v, "", "REPORTED", [measure(e1, k, v, "", level="REPORTED")])
    sub.obj("Body", body.id, origin="body.awaken", epistemic="REPORTED",
            kind="PHYSICAL", device=dev.type, name=body.name)

    # E2. Memory: how much can I really hold? Written, not just reserved.
    e2 = experiment("memory", "how many bytes can this body actually hold and write?")
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        target = int(free * 0.92)
        capped = quick and target > 2 << 30
        if capped:
            target = 2 << 30
        chunk, got, blocks = 1 << 30, 0, []
        t = time.perf_counter()
        try:
            while got + chunk <= target:
                b = torch.empty(chunk, dtype=torch.uint8, device=dev)
                b.fill_(0x5A)
                blocks.append(b)
                got += chunk
            _sync(dev)
        except torch.cuda.OutOfMemoryError:
            pass
        el = time.perf_counter() - t
        del blocks
        torch.cuda.empty_cache()
        m_free = measure(e2, "reported_free", free, "B", level="REPORTED")
        m_got = measure(e2, "written_bytes", got, "B", fill_GBps=got / el / 1e9 if el else None)
        if capped:
            body.usable_bytes = free
            trait("usable_memory", free, "B", "REPORTED", [m_free, m_got],
                  note="quick mode: probe capped at 2 GiB")
        else:
            body.usable_bytes = got
            trait("usable_memory", got, "B", "MEASURED", [m_got])
    else:
        avail = None
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        avail = int(line.split()[1]) * 1024
        except OSError:
            pass
        avail = avail or 4 << 30
        body.usable_bytes = avail // 2
        trait("usable_memory", body.usable_bytes, "B", "REPORTED",
              [measure(e2, "mem_available", avail, "B", level="REPORTED")],
              note="host RAM: half of MemAvailable, not probed")
    log(f"[memory] I can hold {_fmt_bytes(body.usable_bytes)}"
        f" ({sub.q('MATCH (t:Trait {id:$id}) RETURN t.epistemic', id=body.traits['usable_memory'])[0][0]}).")

    # E3. Bandwidth by working-set size; bisect the cliff (XVIII).
    e3 = experiment("bandwidth", "does moving data get slower past some size?")
    maxk = (26 if quick else 30) if dev.type == "cuda" else (24 if quick else 27)
    while (2 << maxk) > body.usable_bytes * 0.5 and maxk > 18:
        maxk -= 1

    def bw(n):
        n = max(4096, n // 4096 * 4096)
        x = torch.empty(n // 4, dtype=torch.float32, device=dev).fill_(1.0)
        y = torch.empty_like(x)
        t = per_call(lambda: y.copy_(x), dev, min_time=0.005 if quick else 0.02)
        del x, y
        gbs = 2 * n / t / 1e9
        return gbs, measure(e3, "copy_bandwidth", gbs, "GB/s", bytes=n, working_set=2 * n)

    sizes = [1 << k for k in range(16, maxk + 1)]
    pts = [(n, *bw(n)) for n in sizes]
    peak_i = max(range(len(pts)), key=lambda i: pts[i][1])
    body.dram_gbs = pts[-1][1]
    trait("peak_bandwidth", pts[peak_i][1], "GB/s", "MEASURED", [pts[peak_i][2]],
          at_working_set=2 * pts[peak_i][0])
    trait("dram_bandwidth", pts[-1][1], "GB/s", "MEASURED", [pts[-1][2]],
          at_working_set=2 * pts[-1][0])
    drops = [(pts[i][1] / pts[i + 1][1], i) for i in range(peak_i, len(pts) - 1)]
    if drops and max(drops)[0] > 1.3:
        ratio, i = max(drops)
        a, b = pts[i], pts[i + 1]
        proof = [a[2], b[2]]
        for _ in range(2 if quick else 4):
            mid = int(math.sqrt(a[0] * b[0]))
            g, m = bw(mid)
            proof.append(m)
            if abs(math.log(g / a[1])) < abs(math.log(g / b[1])):
                a = (mid, g, m)
            else:
                b = (mid, g, m)
        cliff = 2 * int(math.sqrt(a[0] * b[0]))
        trait("cache_cliff", cliff, "B", "INFERRED", proof, drop_ratio=round(ratio, 2),
              meaning="working sets above this lose fast memory (a cache, by inference)")
        log(f"[discovery] bandwidth falls {ratio:.1f}x past ~{_fmt_bytes(cliff)} of working set. "
            f"Something small and fast lives below that line. I bisected {len(proof) - 2} times to find it.")
    log(f"[bandwidth] peak {pts[peak_i][1]:.0f} GB/s, far memory {pts[-1][1]:.0f} GB/s.")

    # E4. Arithmetic: which number formats exist, how fast, how wrong (XXV).
    e4 = experiment("arithmetic", "which number formats work here, how fast and how exact?")
    cands = ["fp32", "tf32", "bf16", "fp16"] if dev.type == "cuda" else ["fp32", "bf16"]
    msizes = ([512, 1024, 2048] if quick else [512, 1024, 2048, 4096, 8192]) if dev.type == "cuda" \
        else ([128, 256] if quick else [128, 256, 512, 1024])
    meter = EnergyMeter(dev)
    body.energy = meter.available
    old_tf32 = torch.backends.cuda.matmul.allow_tf32 if dev.type == "cuda" else None
    for name in cands:
        dt = DTYPES[name]
        try:
            if dev.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = name == "tf32"
            a = torch.randn(256, 256, dtype=torch.float64, device=dev)
            b = torch.randn(256, 256, dtype=torch.float64, device=dev)
            ref = a @ b
            err = float((((a.to(dt) @ b.to(dt)).double() - ref).norm() / ref.norm()).item())
            big = torch.full((64, 64), 300.0, device=dev, dtype=dt)
            range_ok = bool(torch.isfinite(big @ big).all().item())
            proof = [measure(e4, "relative_error", err, "", dtype=name, n=256),
                     measure(e4, "range_300x300x64", range_ok, "", dtype=name)]
            curve = []
            for n in msizes:
                A = torch.randn(n, n, device=dev, dtype=dt)
                B = torch.randn(n, n, device=dev, dtype=dt)
                t = per_call(lambda: A @ B, dev, min_time=0.005 if quick else 0.03)
                tf = 2 * n**3 / t / 1e12
                curve.append((n, tf))
                proof.append(measure(e4, "matmul_tflops", tf, "TFLOPS", dtype=name, n=n))
            watts = None
            if meter.available and not quick:
                with meter:
                    end = time.perf_counter() + 2.0
                    while time.perf_counter() < end:
                        for _ in range(8):
                            A @ B
                        _sync(dev)
                watts = meter.watts
                proof.append(measure(e4, "power_under_matmul", watts, "W", dtype=name, n=msizes[-1]))
            del A, B
            body.dtypes[name] = {"curve": curve, "err": err, "range_ok": range_ok}
            peak = max(t for _, t in curve)
            trait(f"format_{name}", peak, "TFLOPS", "MEASURED", proof, relative_error=err,
                  valid_when="|partial sums| < 65504" if not range_ok else "full fp32 range",
                  tflops_per_watt=peak / watts if watts else None)
            log(f"[arithmetic] {name}: {peak:7.2f} TFLOPS, error {err:.1e}"
                + ("" if range_ok else ", OVERFLOWS above 65504 (valid only inside that envelope)")
                + (f", {peak / watts:.3f} TFLOPS/W" if watts else ""))
        except Exception as ex:  # a format this body lacks is a finding, not a crash
            trait(f"format_{name}", 0, "TFLOPS", "MEASURED",
                  [measure(e4, "unsupported", str(ex)[:120], "", dtype=name)], unsupported=True)
            log(f"[arithmetic] {name}: not available here ({type(ex).__name__}).")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    # E5. Metabolism: power and heat (VI, LII). Absent sensors stay UNKNOWN.
    e5 = experiment("metabolism", "what does thinking cost in watts and heat?")
    if meter.available:
        with meter:
            time.sleep(0.5 if quick else 1.5)
        idle = meter.watts
        temp = meter.temp() if meter.temp else None
        trait("idle_power", idle, "W", "MEASURED", [measure(e5, "idle_power", idle, "W")])
        if temp is not None:
            trait("temperature", temp, "C", "MEASURED", [measure(e5, "temperature", temp, "C")])
        log(f"[metabolism] resting at {idle:.0f} W" + (f", {temp:.0f} C." if temp is not None else "."))
    else:
        trait("energy_sensor", "NOT MEASURED", "", "UNKNOWN",
              [measure(e5, "power_sensor", "absent", "", level="OBSERVED")])
        log("[metabolism] no power sensor in this body: energy is NOT MEASURED, I will use time as cost.")

    trait("self_model_efficiency", body.efficiency, "ratio", "ASSUMED", [],
          meaning="fraction of peak matmul throughput reached while training; corrected by experience")
    sub.event("BODY_KNOWN", body.id, **body.summary())
    return body
