import math
import random
import timeit
import threading
import os
import sys
import multiprocessing
import psutil
import datetime
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


# =====================================================================
# Continuous System Monitor
# =====================================================================

class SystemMonitor(threading.Thread):
    """Monitor CPU load (avg) and clock speed (peak) while running."""
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.cpu_samples = []
        self.peak_freq = 0.0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            cpu_load = psutil.cpu_percent(interval=None)
            freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            self.cpu_samples.append(cpu_load)
            self.peak_freq = max(self.peak_freq, freq)
            timeit.time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

    @property
    def average_cpu(self):
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0

# =====================================================================
# System Metrics (Memory + Battery)
# =====================================================================

def system_metrics_before():
    battery = psutil.sensors_battery()
    return {
        "memory": psutil.virtual_memory().used / (1024 * 1024),
        "battery": battery.percent if battery else None,
        "charging": battery.power_plugged if battery else None,
    }

def system_metrics_after(before):
    battery = psutil.sensors_battery()
    after = {
        "memory": psutil.virtual_memory().used / (1024 * 1024),
        "battery": battery.percent if battery else None,
        "charging": battery.power_plugged if battery else None,
    }

    battery_consumption = None
    if before["battery"] is not None and after["battery"] is not None:
        battery_consumption = before["battery"] - after["battery"]

    return {
        "peak_memory": max(before["memory"], after["memory"]),
        "battery_before": before["battery"],
        "battery_after": after["battery"],
        "battery_consumption": battery_consumption,
        "charging": after["charging"],
    }

# =====================================================================
# Batch Runner Function
# =====================================================================

def run_batch(semiprimes: List[int], save_dir: str):
    before = system_metrics_before()
    if before["charging"]:
        print("⚠️  Device is charging. Skipping benchmark to avoid interference.")
        return
    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    start_time = timeit.default_timer()
    successes = 0

    for n in semiprimes:
        try:
            p, q, used_workers, curves = factorize_semiprime_ecm_parallel(n)
            print(f"✓ Success on {n}")
            successes += 1
        except Exception as e:
            print(f"✗ Failed on {n} ({e})")

    total_runtime = timeit.default_timer() - start_time
    monitor.stop()
    monitor.join()
    after_metrics = system_metrics_after(before_metrics)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Batch Summary")
    print("=" * 70)
    print(f"Total Semiprimes Tested: {len(semiprimes)}")
    print(f"Successful Runs: {successes}/{len(semiprimes)}")
    success_rate = (successes / len(semiprimes)) * 100
    print(f"Success Rate: {success_rate:.2f}%")

    print("\n--- Performance Metrics ---")
    print(f"Total Runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"Throughput: {successes / total_runtime:.4f} successful factorizations/second")
    print(f"Average CPU Load: {monitor.average_cpu:.2f}%")
    print(f"Peak RAM Usage: {after_metrics['peak_memory']:.2f} MB")
    print(f"Peak CPU Clock Speed: {monitor.peak_freq:.2f} MHz")

    if after_metrics["battery_before"] is not None:
        print(f"Battery Before: {after_metrics['battery_before']}%")
        print(f"Battery After: {after_metrics['battery_after']}%")
        print(f"Battery Consumption: {after_metrics['battery_consumption']:.2f}%")

    print("=" * 70)


# ============================================================================
# Benchmark and Aggregate Results
# ============================================================================

def factorize_semiprime_ecm_parallel(n: int, max_curves: int = 4000, B1: int = 100000, workers: int = None) -> Tuple[int, int, int, int]:
    factor, curves, workers = ecm_factor_parallel(n, max_curves=max_curves, B1=B1, workers=workers)
    if factor and factor != n and n % factor == 0:
        return factor, n // factor, workers, curves
    else:
        raise ValueError("ECM failed to factorize")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algorithm_name = "ECM"

    save_dir = r"C:\Users\Frances Bea Magdayao\Desktop\Thesisenv\Objective 2\ECM AW logs"
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"{algorithm_name}_{timestamp}.txt")

    sys.stdout = DualLogger(log_filename)
    print(f"Log started at {timestamp}")
    print(f"Output file: {log_filename}")
    print("=" * 70)
    
    #just need to define 5 semiprimes for testing
    semiprimes = [
        16124098885321590508906039,
        99769848203679794790973289,
        413382272494408950787512431,
        1571660392463995502966917,
        209497331386052445391

    ]

    print(f"Starting ECM Batch Benchmark ({len(semiprimes)} semiprimes)")
    print("=" * 70)

    run_batch(semiprimes, save_dir)

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✅ Output also saved to:\n{log_filename}")
