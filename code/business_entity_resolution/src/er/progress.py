"""Console feedback: timestamped log lines (also tee'd to work/run.log), stage banners with durations,
throttled progress lines with rate + ETA, and a heartbeat for long single calls that cannot report progress.
ASCII only, so it is safe on any Windows console code page."""
import sys
import threading
import time
from contextlib import contextmanager
from .config import WORK_DIR

_T0 = time.time()
_LOGFILE = None


def _fmt(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _ram() -> str:
    try:
        import psutil
        v = psutil.virtual_memory()
        return f" | RAM {v.used / 1e9:.1f}/{v.total / 1e9:.0f}GB"
    except Exception:
        return ""


def log(*a):
    global _LOGFILE
    line = f"[{time.strftime('%H:%M:%S')} +{_fmt(time.time() - _T0)}{_ram()}] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    try:
        if _LOGFILE is None:
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            _LOGFILE = open(WORK_DIR / "run.log", "a", encoding="utf-8")
        _LOGFILE.write(line + "\n")
        _LOGFILE.flush()
    except Exception:
        pass


def banner(title: str):
    bar = "=" * 78
    log("\n" + bar + f"\n  {title}\n" + bar)


@contextmanager
def heartbeat(msg: str, every: float = 20.0):
    """Prints 'still working' every ``every`` seconds while a long call runs."""
    stop = threading.Event()
    t0 = time.time()

    def beat():
        while not stop.wait(every):
            log(f"   ... still {msg} ({_fmt(time.time() - t0)} so far)")

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()


@contextmanager
def stage(title: str, heartbeat_every: float = 20.0):
    """Announces a step, keeps a heartbeat going, and reports its duration."""
    log(f">> {title}")
    t0 = time.time()
    with heartbeat(title, heartbeat_every):
        yield
    log(f"OK {title}  (took {_fmt(time.time() - t0)})")


class Progress:
    """Throttled progress line: 'desc: 34/120 (28.3%) | 1.2k/s | elapsed 00:41 | ETA 01:44 | extra'."""

    def __init__(self, total: int, desc: str, every: float = 5.0, unit: str = ""):
        self.total, self.desc, self.every, self.unit = max(int(total), 1), desc, every, unit
        self.n, self.t0, self.last = 0, time.time(), 0.0

    def update(self, n: int = 1, extra: str = ""):
        self.n += n
        now = time.time()
        if now - self.last >= self.every or self.n >= self.total:
            self.last = now
            el = now - self.t0
            rate = self.n / el if el > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else 0.0
            log(f"   {self.desc}: {self.n:,}/{self.total:,} ({100 * self.n / self.total:.1f}%) | "
                f"{rate:,.0f}{self.unit}/s | elapsed {_fmt(el)} | ETA {_fmt(eta)}" + (f" | {extra}" if extra else ""))


def table(title: str, rows: dict, fmt: str = "{:.4f}"):
    log(title + "\n" + "\n".join(f"      {k!s:>8} : {fmt.format(v)}" for k, v in rows.items()))
