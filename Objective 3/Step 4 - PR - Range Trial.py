import math
import random
import timeit
import multiprocessing
import psutil
import sys
import os
import datetime
import threading
from typing import Optional, Tuple

# =====================================================================
# Dual Output Logger (Console + File)
# =====================================================================

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

# =====================================================================
# Pollard's Rho Algorithm
# =====================================================================

def pollards_rho(n: int, max_attempts: int = 100) -> Optional[int]:
    """Pollard's Rho algorithm for integer factorization."""
    if n % 2 == 0:
        return 2

    for _ in range(max_attempts):
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

# =====================================================================
# Factorization Function
# =====================================================================

def factorize_semiprime(n: int) -> Tuple[int, int]:
    """Attempt to factor a semiprime number using Pollard's Rho"""
    factor = pollards_rho(n)
    if factor and factor != n and n % factor == 0:
        return factor, n // factor
    else:
        raise ValueError("Pollard's Rho failed to factor the number")

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
# Benchmark Function
# =====================================================================

def benchmark_pollards_rho(n: int) -> bool:
    """Run Pollard’s Rho on a semiprime and report only success/fail."""
    success = factorize_semiprime(n)
    if success:
        print(f"✓ Success on {n}")
    else:
        print(f"✗ Failed on {n}")
    return success


def run_batch(semiprimes):
    """Runs Pollard’s Rho on a list of semiprimes and reports batch results."""
    print(f"Starting Pollard’s Rho Batch Benchmark ({len(semiprimes)} semiprimes)")
    print("=" * 70)

    before_metrics = system_metrics_before()
    if before_metrics["charging"]:
        print("⚠️  Device is charging. Skipping benchmark.")
        return

    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    start_time = timeit.default_timer()
    successes = 0

    for n in semiprimes:
        if benchmark_pollards_rho(n):
            successes += 1

    total_runtime = timeit.default_timer() - start_time
    monitor.stop()
    monitor.join()
    after_metrics = system_metrics_after(before_metrics)

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


# =====================================================================
# Main Runner
# =====================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # --- Setup Log Directory and File ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algorithm_name = "PollardsRho"
    save_dir = r"C:\Users\Frances Bea Magdayao\Desktop\Thesisenv\Objective 2\PR AW logs"
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"{algorithm_name}_{timestamp}.txt")

    # --- Start Dual Logging ---
    sys.stdout = DualLogger(log_filename)
    print(f"Log started at {timestamp}")
    print(f"Output file: {log_filename}")
    print("=" * 70)

    # add 2 semiprime for 5 Trial Test
    semiprimes = [
        12942855675206633208055721,
        203830917044970189650534581,
        693440746629641342130899

    ]

    run_batch(semiprimes)

    # --- Restore Normal Output ---
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✅ Output also saved to:\n{log_filename}")
