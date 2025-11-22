import math
import random
import timeit
import multiprocessing
import psutil
import sys
import os
import datetime
import threading
import time
from typing import Optional, Tuple


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
# Pollard's Rho Core
# ============================================================================

def pollards_rho(n: int, max_attempts: int = 100) -> Optional[int]:
    """Pollard's Rho algorithm for integer factorization"""
    if n % 2 == 0:
        return 2

    for attempt in range(max_attempts):
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        f = lambda x: (pow(x, 2, n) + c) % n
        d = 1

        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
            if d == n:
                break

        if d != n and d != 1:
            return d

    return None


# ============================================================================
# Factorization
# ============================================================================

def factorize_semiprime(n: int) -> Tuple[int, int]:
    """Attempt to factor a semiprime number using Pollard's Rho"""
    factor = pollards_rho(n)
    if factor and factor != n and n % factor == 0:
        return factor, n // factor
    else:
        raise ValueError("Pollard's Rho failed to factor the number")


# ============================================================================
# System Metrics Sampler (Average CPU + Peak Frequency)
# ============================================================================

class SystemMonitor(threading.Thread):
    """Monitors system-wide CPU load and clock speed during computation."""
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
        """Compute average CPU load from samples."""
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


# ============================================================================
# Benchmark Function
# ============================================================================

def benchmark_pollards_rho(n: int) -> None:
    """Single run benchmark with system metrics"""
    print(f"Factoring {n} using Pollard's Rho")
    print(f"Bit length: {n.bit_length()} bits")
    print("=" * 60)

    battery = psutil.sensors_battery()
    before_battery = battery.percent if battery else None
    if battery and battery.power_plugged:
        print("⚠️  Device is charging. Skipping benchmark to avoid battery interference.")
        return

    before_memory = psutil.virtual_memory().used / (1024 * 1024)

    # --- Start monitoring thread ---
    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    start_time = timeit.default_timer()
    try:
        p, q = factorize_semiprime(n)
    finally:
        monitor.stop()
        monitor.join()
    end_time = timeit.default_timer()

    after_memory = psutil.virtual_memory().used / (1024 * 1024)
    after_battery = psutil.sensors_battery().percent if psutil.sensors_battery() else None

    runtime = end_time - start_time

    # --- Factor verification ---
    if p * q == n:
        print(f"Result: {n} = {p} × {q}")
        print(f"Verified: ✓ Success")
    else:
        print(f"Result: {p} × {q} = {p * q}")
        print(f"Verified: ✗ Failed (incorrect factors)")

    # --- Metrics ---
    avg_cpu = monitor.average_cpu()
    peak_mem = max(before_memory, after_memory)

    print("\n--- Performance Metrics ---")
    print(f"Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    print(f"Average CPU Load: {avg_cpu:.2f}%")
    print(f"Peak CPU Clock Speed: {monitor.peak_freq:.2f} MHz")
    print(f"Peak RAM Usage: {peak_mem:.2f} MB")

    if before_battery is not None and after_battery is not None:
        consumption = before_battery - after_battery
        print(f"Battery Before: {before_battery}%")
        print(f"Battery After: {after_battery}%")
        print(f"Battery Consumption: {consumption:.2f}%")

    print("=" * 60)


# ============================================================================
# Main Runner
# ============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # --- Setup Log Directory and File ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algorithm_name = "PollardsRho_AvgMetrics"

    save_dir = r"C:\Users\acer\Desktop\Non-Linear-Regression-of-PR-QS-ECM\Objective 3\Step 4 Resources\PR_Logs"
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"{algorithm_name}_{timestamp}.txt")

    # Start dual logging
    sys.stdout = DualLogger(log_filename)

    print(f"Log started at {timestamp}")
    print(f"Output file: {log_filename}")
    print("=" * 70)

    n = int(input("Enter a semiprime to factor: "))
    benchmark_pollards_rho(n)

    # Restore stdout
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

    print(f"\n✅ Output also saved to:\n{log_filename}")
