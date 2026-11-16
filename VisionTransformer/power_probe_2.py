#!/usr/bin/env python
"""
power_probe_2.py — what power/energy telemetry can a user read from inside a MIG slice?

power_metrics.py showed that nvmlDeviceGetPowerUsage on the visible handle is the
power of the whole B200 board. This probe checks the remaining user-side options:

  1. which physical card the slice is on (parent GPU UUID), so two jobs can tell
     whether they share a card;
  2. NVML queries not tried before: the board's cumulative energy counter, the
     instant/average power field values, and power on the MIG device handle itself;
  3. whether the board signal is usable by subtraction: power sampled at 10 Hz in
     idle / load / idle phases (load = fp32 matmuls on our slice), repeated. If the
     idle phases are flat and the load phases clearly above them, our share can be
     estimated as (load - idle) while the other slices stay quiet.

Every query is guarded; the probe always reaches its summary. Writes a JSON next to
the .out file (--out).
"""
import argparse
import json
import socket
import os
import threading
import time

import pynvml
import torch


def safe(label, fn, results):
    try:
        v = fn()
        results[label] = v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)
        print(f"{label:<34}: {results[label]}")
    except Exception as exc:  # noqa: BLE001
        results[label] = f"UNAVAILABLE ({type(exc).__name__}: {exc})"
        print(f"{label:<34}: {results[label]}")


def _s(v):
    return v.decode() if isinstance(v, bytes) else v


def sample_power(h, seconds, hz=10):
    """Board power (W) and energy counter (J) sampled at `hz` for `seconds`."""
    p, t = [], []
    e0 = _energy(h)
    end = time.time() + seconds
    while time.time() < end:
        try:
            p.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            t.append(time.time())
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0 / hz)
    e1 = _energy(h)
    p_sorted = sorted(p) or [float("nan")]
    return {"n": len(p), "mean_w": sum(p) / max(1, len(p)), "min_w": p_sorted[0], "max_w": p_sorted[-1],
            "p50_w": p_sorted[len(p_sorted) // 2],
            "energy_counter_j": (e1 - e0) if (e0 is not None and e1 is not None) else None,
            "seconds": seconds}


def _energy(h):
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0  # mJ -> J
    except Exception:  # noqa: BLE001
        return None


def load(seconds, stop):
    """Steady fp32 matmul load on our slice."""
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    end = time.time() + seconds
    while time.time() < end and not stop.is_set():
        for _ in range(10):
            a = (a @ b) * 1e-4
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-seconds", type=int, default=30)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--out", default="power_probe_2.json")
    args = ap.parse_args()

    r = {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "queries": {}}
    q = r["queries"]
    pynvml.nvmlInit()
    safe("driver", lambda: _s(pynvml.nvmlSystemGetDriverVersion()), q)
    safe("visible device count", pynvml.nvmlDeviceGetCount, q)
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    safe("name", lambda: _s(pynvml.nvmlDeviceGetName(h)), q)
    safe("parent GPU UUID", lambda: _s(pynvml.nvmlDeviceGetUUID(h)), q)
    safe("PCI bus id", lambda: _s(pynvml.nvmlDeviceGetPciInfo(h).busId), q)
    safe("memory total (GB)", lambda: pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e9, q)
    safe("MIG mode (current, pending)", lambda: list(pynvml.nvmlDeviceGetMigMode(h)), q)
    safe("power usage (W)", lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, q)
    safe("power limit (W)", lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0, q)
    safe("total energy counter (J)", lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0, q)
    for name in ("NVML_FI_DEV_POWER_INSTANT", "NVML_FI_DEV_POWER_AVERAGE"):
        fid = getattr(pynvml, name, None)
        if fid is None:
            q[name] = "UNAVAILABLE (not in this pynvml)"
            print(f"{name:<34}: {q[name]}")
            continue
        safe(name, lambda fid=fid: [(_v.nvmlReturn, _v.value.uiVal)
                                    for _v in pynvml.nvmlDeviceGetFieldValues(h, [fid])], q)
    safe("utilization", lambda: str(pynvml.nvmlDeviceGetUtilizationRates(h)), q)
    safe("running processes visible", lambda: len(pynvml.nvmlDeviceGetComputeRunningProcesses(h)), q)
    # the MIG device handle(s) we can see, and whether they report power of their own
    try:
        n = pynvml.nvmlDeviceGetMaxMigDeviceCount(h)
        q["max MIG devices"] = n
        for i in range(n):
            try:
                mh = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(h, i)
            except Exception:  # noqa: BLE001
                continue
            safe(f"MIG[{i}] UUID", lambda: _s(pynvml.nvmlDeviceGetUUID(mh)), q)
            safe(f"MIG[{i}] memory (GB)", lambda: pynvml.nvmlDeviceGetMemoryInfo(mh).total / 1e9, q)
            safe(f"MIG[{i}] power usage (W)", lambda: pynvml.nvmlDeviceGetPowerUsage(mh) / 1000.0, q)
            safe(f"MIG[{i}] energy counter (J)",
                 lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(mh) / 1000.0, q)
    except Exception as exc:  # noqa: BLE001
        q["MIG handles"] = f"UNAVAILABLE ({type(exc).__name__}: {exc})"

    # idle / load / idle on the board signal
    print("\n[probe] idle / load cycles on the board power signal")
    phases = []
    try:
        torch.zeros(1, device="cuda")
        for c in range(args.cycles):
            phases.append({"cycle": c, "phase": "idle", **sample_power(h, args.phase_seconds)})
            stop = threading.Event()
            th = threading.Thread(target=load, args=(args.phase_seconds + 2, stop))
            th.start()
            time.sleep(1)
            phases.append({"cycle": c, "phase": "load", **sample_power(h, args.phase_seconds)})
            stop.set()
            th.join()
            torch.cuda.synchronize()
        phases.append({"cycle": args.cycles, "phase": "idle", **sample_power(h, args.phase_seconds)})
    except Exception as exc:  # noqa: BLE001
        phases.append({"error": f"{type(exc).__name__}: {exc}"})
    r["phases"] = phases
    for p in phases:
        if "mean_w" in p:
            print(f"  cycle {p['cycle']} {p['phase']:<5} mean {p['mean_w']:7.1f} W  "
                  f"min {p['min_w']:7.1f}  max {p['max_w']:7.1f}  energy {p['energy_counter_j']}")
    pynvml.nvmlShutdown()
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
