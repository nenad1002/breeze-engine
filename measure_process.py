#!/usr/bin/env python3
"""Linux child-process CPU/RSS sampling with authoritative wait4 accounting.

Usage: measure_process.py --label NAME [--interval SECONDS] -- COMMAND [ARGS ...]
The command is executed as explicit argv, never through a shell; the parent
waits and propagates its exit code (128 + signal number for signal termination).
CPU percentages are process-wide: 4800 means 48 cores fully busy. CPU%_busy is
the median of the upper half of interval samples, only a heuristic: it does NOT
identify a decode phase. Tail RSS likewise describes time, not a model phase.
SIGINT stops/reaps only the direct child and stops the sampler; no process-group
or unrelated-process signals are sent. Descendant processes are not managed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
import signal
from statistics import median
import sys
import threading
from time import perf_counter


def positive_interval(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= threading.TIMEOUT_MAX:
        raise argparse.ArgumentTypeError("interval must be finite, positive and within the timer limit")
    return value


def read_cpu_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as stream:
            data = stream.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    rest = data[data.rfind(")") + 2:].split()  # comm may contain spaces or ')'
    return int(rest[11]) + int(rest[12])  # utime + stime; malformed data is an error


def read_rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None


def sample_process(pid, stop, interval, start, hz, *, clock=perf_counter):
    """Sample until signalled; Event.wait permits immediate shutdown."""
    samples = []
    previous = None
    while not stop.is_set():
        now = clock()
        ticks = read_cpu_ticks(pid)
        rss = read_rss_kb(pid)
        if ticks is not None and previous is not None:
            delta = now - previous[0]
            if delta > 0:
                percent = 100 * (ticks - previous[1]) / (hz * delta)
                samples.append((now - start, percent, rss or 0))
        if ticks is not None:
            previous = (now, ticks)
        stop.wait(interval)
    return samples


def exit_code(status):
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code


def wait_child(pid):
    """Blocking wait, retrying only interrupted system calls (not polling)."""
    while True:
        try:
            return os.wait4(pid, 0)
        except InterruptedError:
            continue


def terminate_child(pid):
    """Reap an owned, not-yet-reaped child on interruption/error."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wait_child(pid)
    finally:
        signal.signal(signal.SIGINT, previous)


def measure(command, *, label, interval):
    hz = os.sysconf("SC_CLK_TCK")
    start = perf_counter()
    stop = threading.Event()
    pid = os.fork()
    if pid == 0:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            os.execvp(command[0], command)
        except OSError as exc:
            message = f"Could not execute {command[0]!r}: {exc}\n".encode(errors="replace")
            try:
                os.write(2, message)
            finally:
                os._exit(127 if isinstance(exc, FileNotFoundError) else 126)

    reaped = False
    executor = None
    try:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="process-sampler")
        sampler = executor.submit(sample_process, pid, stop, interval, start, hz)
        _, status, usage = wait_child(pid)
        reaped = True
        wall = perf_counter() - start
        stop.set()
        # Surface sampler programming errors rather than silently losing samples.
        samples = sampler.result()
    except KeyboardInterrupt:
        return 130
    finally:
        stop.set()
        try:
            if not reaped:
                terminate_child(pid)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    cpu_total = usage.ru_utime + usage.ru_stime
    peak_gib = usage.ru_maxrss / (1024 * 1024)  # Linux reports KiB
    whole_pct = 100 * cpu_total / wall if wall > 0 else 0.0
    busy = sorted(percent for _, percent, _ in samples)
    busy_text = f"{median(busy[len(busy) // 2:]):.0f}" if busy else "n/a"
    tail = [rss for elapsed, _, rss in samples if elapsed > 0.75 * wall and rss > 0]
    tail_text = f"{median(tail) / (1024 * 1024):.3f}GiB" if tail else "n/a"
    code = exit_code(status)
    print(f"\n##MEASURE## {label} wall={wall:.3f}s CPU%_whole={whole_pct:.0f} "
          f"CPU%_busy={busy_text} (upper-half heuristic) peakRSS={peak_gib:.3f}GiB "
          f"tailRSS={tail_text} samples={len(samples)} exit={code}", file=sys.stderr)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--interval", type=positive_interval, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS ...]")
    args = parser.parse_args(argv)
    if not args.label.strip():
        parser.error("--label must not be empty")
    if not args.command or args.command[0] != "--" or len(args.command) < 2 or not args.command[1]:
        parser.error("supply an explicit command after --")
    if not sys.platform.startswith("linux") or not hasattr(os, "wait4"):
        parser.error("process sampling requires Linux and os.wait4")
    return measure(args.command[1:], label=args.label, interval=args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
