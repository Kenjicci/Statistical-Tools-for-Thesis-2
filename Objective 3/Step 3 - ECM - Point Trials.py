import math
import random
import timeit
import os
import sys
import multiprocessing
import psutil
import datetime
import threading
import time
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================================
# Dual Output Logger (Console + File)
# ============================================================================

class DualLogger:
    """Redirects print() output to both console and a log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ============================================================================
# Utility Function
# ============================================================================

def mod_inverse(a: int, m: int) -> Optional[int]:
    try:
        return pow(a, -1, m)
    except ValueError:
        return None

def sieve_primes(limit: int) -> List[int]:
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]


# ============================================================================
# ECM Core
# ============================================================================

class ECMPoint:
    def __init__(self, x: int, z: int, n: int):
        self.x = x % n
        self.z = z % n
        self.n = n

    def __repr__(self):
        return f"ECMPoint({self.x}, {self.z})"


def ecm_add(P: ECMPoint, Q: ECMPoint, diff: ECMPoint, a24: int) -> ECMPoint:
    n = P.n
    u = ((P.x - P.z) * (Q.x + Q.z)) % n
    v = ((P.x + P.z) * (Q.x - Q.z)) % n
    upv = (u + v) % n
    umv = (u - v) % n
    x = (diff.z * upv * upv) % n
    z = (diff.x * umv * umv) % n
    return ECMPoint(x, z, n)


def ecm_double(P: ECMPoint, a24: int) -> ECMPoint:
    n = P.n
    u = (P.x + P.z) % n
    v = (P.x - P.z) % n
    u2 = (u * u) % n
    v2 = (v * v) % n
    x = (u2 * v2) % n
    diff = (u2 - v2) % n
    z = (diff * ((v2 + a24 * diff) % n)) % n
    return ECMPoint(x, z, n)


def ecm_multiply(k: int, P: ECMPoint, a24: int) -> ECMPoint:
    R0 = ECMPoint(1, 0, P.n)
    R1 = P
    for bit in bin(k)[2:]:
        if bit == '0':
            R1 = ecm_add(R0, R1, P, a24)
            R0 = ecm_double(R0, a24)
        else:
            R0 = ecm_add(R0, R1, P, a24)
            R1 = ecm_double(R1, a24)
    return R0


def generate_curve_and_point(n: int) -> Tuple[ECMPoint, int]:
    while True:
        sigma = random.randint(6, n - 1)
        u = (sigma * sigma - 5) % n
        v = (4 * sigma) % n
        inv_v = mod_inverse(v, n)
        if not inv_v:
            continue
        x = (u ** 3 * inv_v) % n
        P = ECMPoint(x, 1, n)
        try:
            a24 = ((random.randint(2, n-1) + 2) * mod_inverse(4, n)) % n
            return P, a24
        except Exception:
            continue


def ecm_stage1(P: ECMPoint, a24: int, B1: int) -> Tuple[ECMPoint, List[int]]:
    n = P.n
    primes = sieve_primes(B1)
    Q = P
    for p in primes:
        power = p
        while power * p <= B1:
            power *= p
        Q = ecm_multiply(power, Q, a24)
        g = math.gcd(Q.z, n)
        if 1 < g < n:
            return Q, primes
    return Q, primes


def ecm_factor_single(n: int, max_curves: int, B1: int, seed: int) -> Optional[int]:
    random.seed(seed)
    for _ in range(max_curves):
        P, a24 = generate_curve_and_point(n)
        Q, _ = ecm_stage1(P, a24, B1)
        g = math.gcd(Q.z, n)
        if 1 < g < n:
            return g
    return None


# ============================================================================
# Parallel Factorization
# ============================================================================

def ecm_factor_parallel(n: int, max_curves: int = 4000, B1: int = 100000, workers: int = None) -> Tuple[Optional[int], int, int]:
    """Return (factor, curves_attempted, workers_used)."""
    if workers is None:
        workers = os.cpu_count()
        
    workers = max(1, workers)
    curves_per_worker = max(1, max_curves // workers)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(ecm_factor_single, n, curves_per_worker, B1, i + 1)
            for i in range(workers)
        ]

        for future in as_completed(futures):
            try:
                factor = future.result()
            except Exception:
                factor = None
            if factor:
                executor.shutdown(cancel_futures=True)
                return factor, max_curves, workers

    return None, max_curves, workers


# ============================================================================
# System Monitor (Average CPU + Peak Clock Speed)
# ============================================================================

class SystemMonitor(threading.Thread):
    """Monitors average CPU load and peak clock speed during execution."""
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.samples = []
        self.peak_freq = 0.0
        self._stop_event = threading.Event()

    def run(self):
        psutil.cpu_percent(interval=None)  # Initialize baseline
        while not self._stop_event.is_set():
            cpu_load = psutil.cpu_percent(interval=None)
            freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            self.samples.append(cpu_load)
            self.peak_freq = max(self.peak_freq, freq)
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

    def average_cpu(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


# ============================================================================
# Benchmark Wrapper
# ============================================================================

def factorize_semiprime_ecm_parallel(n: int, max_curves: int = 4000, B1: int = 100000, workers: int = None) -> Tuple[int, int, int, int]:
    """Return (p, q, workers_used, curves_attempted)."""
    factor, curves, workers = ecm_factor_parallel(n, max_curves=max_curves, B1=B1, workers=workers)
    if factor and factor != n and n % factor == 0:
        return factor, n // factor, workers, curves
    else:
        raise ValueError("ECM failed to factorize")


def benchmark_ecm_parallel(n: int, max_curves: int = 4000, B1: int = 100000, workers: int = None):
    bit_size = n.bit_length()
    
    print(f"\n{'='*70}")
    print(f"Factoring with Parallel ECM (Stage 1 Only)")
    print(f"{'='*70}")
    print(f"Number: {n}")
    print(f"Bit size: {bit_size} bits")
    print(f"B1 (Stage 1 bound): {B1:,}")
    print(f"Max curves: {max_curves:,}")
    print(f"Workers: {workers or os.cpu_count()}")
    print(f"{'='*70}\n")

    battery = psutil.sensors_battery()
    before_battery = battery.percent if battery else None
    if battery and battery.power_plugged:
        print("⚠️  Device is charging. Skipping benchmark to avoid battery interference.")
        return

    before_memory = psutil.virtual_memory().used / (1024 * 1024)

    # Start system monitor
    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    start_time = timeit.default_timer()
    try:
        p, q, used_workers, curves = factorize_semiprime_ecm_parallel(n, max_curves=max_curves, B1=B1, workers=workers)
    finally:
        monitor.stop()
        monitor.join()
    end_time = timeit.default_timer()

    after_memory = psutil.virtual_memory().used / (1024 * 1024)
    after_battery = psutil.sensors_battery().percent if psutil.sensors_battery() else None

    runtime = end_time - start_time

    # --- Verification ---
    if p * q == n:
        print(f"Result: {n} = {p} × {q}")
        print(f"Verified: ✓ Success")
    else:
        print(f"Result: {p} × {q} = {p * q}")
        print(f"Verified: ✗ Failed (incorrect factors)")

    print(f"Workers used: {used_workers}")
    print(f"Curves attempted: ~{curves:,}")

    avg_cpu = monitor.average_cpu()
    peak_freq = monitor.peak_freq
    peak_memory = max(before_memory, after_memory)

    print("\n--- Performance Metrics ---")
    print(f"Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    print(f"Average CPU Load: {avg_cpu:.2f}%")
    print(f"Peak CPU Clock Speed: {peak_freq:.2f} MHz")
    print(f"Peak RAM Usage: {peak_memory:.2f} MB")

    if before_battery is not None and after_battery is not None:
        consumption = before_battery - after_battery
        print(f"Battery Before: {before_battery}%")
        print(f"Battery After: {after_battery}%")
        print(f"Battery Consumption: {consumption:.2f}%")

    print("=" * 70)


# ============================================================================
# Main Runner
# ============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # --- Setup Log Directory and File ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algorithm_name = "ECM_AvgMetrics"

    save_dir = r"C:\Users\acer\Desktop\Non-Linear-Regression-of-PR-QS-ECM\Objective 3\Step 4 Resources\ECM_Logs"
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"{algorithm_name}_{timestamp}.txt")

    sys.stdout = DualLogger(log_filename)

    print(f"Log started at {timestamp}")
    print(f"Output file: {log_filename}")
    print("=" * 70)

    n = int(input("Enter a semiprime to factor using ECM: "))
    benchmark_ecm_parallel(n, max_curves=4000, B1=100000, workers=None)

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

    print(f"\n✅ Output also saved to:\n{log_filename}")
