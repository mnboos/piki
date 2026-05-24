"""System metrics for the RDK X5 board.

What's *actually* exposed by the Horizon kernel + standard Linux on this SoC:

| Metric          | Source                                                     |
|---|---|
| CPU load %      | /proc/stat (deltas)                                        |
| CPU freq        | /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq     |
| CPU temp        | /sys/class/thermal/thermal_zone1/temp                      |
| DDR temp        | /sys/class/thermal/thermal_zone0/temp                      |
| BPU temp        | hrut_somstatus (or `bpu` thermal in some images)           |
| BPU load %      | /sys/devices/system/bpu/bpu0/ratio                         |
| BPU freq        | /sys/class/devfreq/3a000000.bpu/cur_freq                   |
| GPU GC8000 freq | /sys/class/devfreq/3c000000.gc8000/cur_freq                |
| DDR freq        | /sys/class/devfreq/soc:ddrc-freq/cur_freq                  |
| VPU clock freq  | /sys/module/hobot_vpu/parameters/vpu_clk_freq              |
| RAM             | /proc/meminfo (psutil)                                     |
| Disk            | psutil.disk_usage('/')                                     |
| Net throughput  | /proc/net/dev deltas                                       |
| Uptime          | /proc/uptime                                               |

**Not** exposed by the SDK (we report the static info only and flag the
percentage as unavailable):

- VPU encode/decode utilisation percentage
- ISP/VPS pipeline utilisation percentage
- GPU GC8000 utilisation percentage

For the VPU, the closest proxy we can offer is the encoder's actual output
frame rate (drives a "VPU activity" indicator).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

# Cumulative net-throughput state so we can compute deltas across calls.
_last_net_sample: tuple[float, int, int] | None = None


def _read_int(path: str | Path) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_temp(path: str) -> float | None:
    """Read a thermal_zone temperature; sysfs reports milli-Celsius."""
    raw = _read_int(path)
    return raw / 1000.0 if raw is not None else None


def _cpu_freqs_mhz() -> list[float]:
    out: list[float] = []
    for i in range(psutil.cpu_count(logical=True) or 0):
        hz = _read_int(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
        # scaling_cur_freq is in kHz; some kernels report 0 if userspace lacks perms.
        out.append((hz or 0) / 1000.0)
    return out


def _devfreq_mhz(name: str) -> float | None:
    hz = _read_int(f"/sys/class/devfreq/{name}/cur_freq")
    return hz / 1_000_000.0 if hz else None


def _bpu_ratio() -> int | None:
    """BPU 0 utilisation percentage (0–100)."""
    return _read_int("/sys/devices/system/bpu/bpu0/ratio")


def _vpu_clock_mhz() -> float | None:
    hz = _read_int("/sys/module/hobot_vpu/parameters/vpu_clk_freq")
    return hz / 1_000_000.0 if hz else None


def _net_throughput() -> tuple[float, float]:
    """Bytes/s rx, tx across all non-loopback interfaces, smoothed over the
    interval between calls. Returns (0, 0) on the first call."""
    global _last_net_sample
    now = time.monotonic()
    total_rx = 0
    total_tx = 0
    counters = psutil.net_io_counters(pernic=True)
    for iface, c in counters.items():
        if iface == "lo":
            continue
        total_rx += c.bytes_recv
        total_tx += c.bytes_sent
    if _last_net_sample is None:
        _last_net_sample = (now, total_rx, total_tx)
        return 0.0, 0.0
    dt = max(1e-3, now - _last_net_sample[0])
    rx_per_s = (total_rx - _last_net_sample[1]) / dt
    tx_per_s = (total_tx - _last_net_sample[2]) / dt
    _last_net_sample = (now, total_rx, total_tx)
    return max(0.0, rx_per_s), max(0.0, tx_per_s)


def collect() -> dict:
    """One-shot snapshot of all reachable system metrics."""
    # cpu_percent(interval=None) requires a prior call to be meaningful; we
    # call it once with interval=None and accept that the first call after
    # import returns 0.
    cpu_percent_per_core = psutil.cpu_percent(interval=None, percpu=True)
    cpu_percent_total = sum(cpu_percent_per_core) / max(1, len(cpu_percent_per_core))

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")

    rx_bps, tx_bps = _net_throughput()

    uptime_s: float | None = None
    try:
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
    except OSError:
        pass

    # Temperatures: thermal_zone0=DDR, thermal_zone1=CPU on this image.
    # BPU temperature is reported via hrut_somstatus but isn't reachable via
    # sysfs on the X5 image (different kernel build than X3). Best effort:
    # try every thermal zone and label by `type`.
    temps: dict[str, float] = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            t_type = (zone / "type").read_text().strip()
            t_val = _read_temp(str(zone / "temp"))
        except OSError:
            continue
        if t_val is not None:
            # Strip the leading 'thermal-' some Horizon images use.
            label = t_type.removeprefix("thermal-")
            temps[label] = round(t_val, 1)

    return {
        "ts": time.time(),
        "uptime_s": uptime_s,
        "load_avg": list(psutil.getloadavg()),
        "cpu": {
            "percent_total": round(cpu_percent_total, 1),
            "percent_per_core": [round(p, 1) for p in cpu_percent_per_core],
            "freq_mhz_per_core": _cpu_freqs_mhz(),
            "core_count": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "used_bytes": vm.used,
            "percent": vm.percent,
        },
        "swap": {
            "total_bytes": swap.total,
            "used_bytes": swap.used,
            "percent": swap.percent,
        },
        "disk_root": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "percent": disk.percent,
        },
        "net": {
            "rx_bytes_per_s": round(rx_bps, 1),
            "tx_bytes_per_s": round(tx_bps, 1),
        },
        "temps_c": temps,
        "bpu": {
            "load_percent": _bpu_ratio(),
            "freq_mhz": _devfreq_mhz("3a000000.bpu"),
        },
        "vpu": {
            # Encoder utilisation isn't exposed by the SDK. Surface only the
            # static clock; the frontend uses webrtc_active + measured encode
            # fps from elsewhere as a proxy.
            "clock_mhz": _vpu_clock_mhz(),
            "load_percent": None,
        },
        "gpu": {
            "freq_mhz": _devfreq_mhz("3c000000.gc8000"),
            "load_percent": None,
        },
        "ddr": {
            "freq_mhz": _devfreq_mhz("soc:ddrc-freq"),
        },
        "isp": {
            "load_percent": None,
        },
    }
