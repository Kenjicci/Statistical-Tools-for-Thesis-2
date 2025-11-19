"""
Quadratic Sieve Integer Factorization Algorithm (Batch Version)
Includes: Performance Monitoring, Average CPU Load, Peak Clock Speed, and Throughput
"""

import math
import timeit
import time
import traceback
import threading
import multiprocessing
import psutil
import sys
import datetime
import os
from math import isqrt
from collections import defaultdict, namedtuple
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import numpy as np

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

# ============================================================================
# Data Structures
# ============================================================================

Relation = namedtuple("Relation", ["x", "factors", "Qx"])

# ============================================================================
# Prime Number and Modular Arithmetic Functions
# ============================================================================

def sieve_of_eratosthenes(limit: int) -> List[int]:
    """
    Generate all prime numbers up to the given limit using the Sieve of Eratosthenes.
    
    Args:
        limit: Upper bound for prime generation
        
    Returns:
        List of prime numbers up to limit
    """
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(2, limit + 1) if sieve[i]]


def legendre_symbol(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a/p).
    Returns 1 if a is a quadratic residue mod p, -1 otherwise.
    
    Args:
        a: Number to test
        p: Prime modulus
        
    Returns:
        Legendre symbol value
    """
    if p == 2:
        return 1
    return pow(a, (p - 1) // 2, p)


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """
    Find square root of n modulo p using the Tonelli-Shanks algorithm.
    
    Args:
        n: Number to find square root of
        p: Prime modulus
        
    Returns:
        Square root of n mod p if it exists, None otherwise
    """
    if legendre_symbol(n, p) != 1:
        return None
    
    # Find Q and S such that p - 1 = Q * 2^S with Q odd
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    if S == 1:
        return pow(n, (p + 1) // 4, p)
    
    # Find quadratic non-residue z
    z = 2
    while legendre_symbol(z, p) != p - 1:
        z += 1
    
    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)
    
    while t != 1:
        # Find smallest i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        # Update variables
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p
    
    return R

# ============================================================================
# Factor Base Generation
# ============================================================================

def factor_base_primes(N: int, bound: int) -> List[int]:
    """
    Generate factor base: primes p <= bound where N is a quadratic residue mod p.
    
    Args:
        N: Number to factor
        bound: Upper bound for primes in factor base
        
    Returns:
        List of primes suitable for factoring N
    """
    primes = sieve_of_eratosthenes(bound)
    factor_base = [-1]  # Include -1 for handling negative numbers
    
    for p in primes:
        if p == 2 or legendre_symbol(N, p) == 1:
            factor_base.append(p)
    
    return factor_base

# ============================================================================
# Factorization Functions
# ============================================================================

def trial_division(n: int, factor_base: List[int]) -> Optional[Dict[int, int]]:
    """
    Factor n completely using trial division with the factor base.
    
    Args:
        n: Number to factor
        factor_base: List of primes to use for factoring
        
    Returns:
        Dictionary of prime factors and exponents, or None if n is not B-smooth
    """
    if n == 0:
        return None
    
    factors = {}
    if n < 0:
        factors[-1] = 1
        n = -n
    
    for p in factor_base:
        if p == -1:
            continue
        
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        
        if count > 0:
            factors[p] = count
        
        if n == 1:
            break
    
    return factors if n == 1 else None

# ============================================================================
# Sieving Functions
# ============================================================================

def sieve_at_offset(args: Tuple) -> List[Relation]:
    """
    Perform sieving at a specific offset range.
    
    Args:
        args: Tuple containing (N, M, offset, factor_base, log_p_cache)
        
    Returns:
        List of smooth relations found
    """
    N, M, offset, factor_base, log_p_cache = args
    sqrtN = isqrt(N)
    start_x = -M + offset
    end_x = M + offset
    sieve_size = end_x - start_x + 1
    
    # Initialize sieve array
    sieve_array = [0.0] * sieve_size
    Qx_list = []
    
    for i in range(sieve_size):
        x = start_x + i
        Qx = (sqrtN + x)**2 - N
        Qx_list.append(Qx)
        
        if Qx > 0:
            sieve_array[i] = math.log(Qx)
        elif Qx < 0:
            sieve_array[i] = math.log(-Qx)
        else:
            sieve_array[i] = float('inf')
    
    # Sieve with each prime in the factor base
    for p in factor_base:
        if p == -1:
            continue
        
        log_p = log_p_cache.get(p, math.log(abs(p)))
        
        if p == 2:
            # Special case for p = 2
            for i in range(sieve_size):
                Qx = Qx_list[i]
                if Qx != 0 and abs(Qx) % p == 0:
                    temp_q = abs(Qx)
                    while temp_q % p == 0:
                        sieve_array[i] -= log_p
                        temp_q //= p
        else:
            # General case for odd primes
            root = tonelli_shanks(N % p, p)
            if root is None:
                continue
            
            roots = [root, p - root] if root != p - root else [root]
            
            for r in roots:
                target = (r - sqrtN) % p
                if target > p // 2:
                    target -= p
                
                x = target
                while x < start_x:
                    x += p
                
                while x <= end_x:
                    i = x - start_x
                    Qx = Qx_list[i]
                    if Qx != 0:
                        temp_q = abs(Qx)
                        while temp_q % p == 0:
                            sieve_array[i] -= log_p
                            temp_q //= p
                    x += p
    
    # Collect smooth relations
    relations = []
    for i in range(sieve_size):
        threshold = math.log(abs(Qx_list[i])) * 0.7
        if sieve_array[i] < threshold:
            x = start_x + i
            Qx = Qx_list[i]
            if Qx != 0:
                factors = trial_division(Qx, factor_base)
                if factors is not None:
                    relations.append(Relation(x, factors, Qx))
    
    return relations


def optimized_sieve_incremental(
    N: int, 
    M: int, 
    factor_base: List[int], 
    log_p_cache: Dict[int, float],
    max_relations: int = 1000,
    chunk_limit: int = 200,
    workers: Optional[int] = None
) -> List[Relation]:
    """
    Perform incremental sieving with multiprocessing.
    """
    collected = []
    offset_step = M * 2
    offset = 0

    if workers is None:
        workers = max(1, cpu_count() - 1)

    print(f"    Sieving incrementally (max {max_relations} relations, {workers} workers)")

    while len(collected) < max_relations:
        offsets = [offset + i * offset_step for i in range(-2, 3)]
        args_list = [(N, M, o, factor_base, log_p_cache) for o in offsets]

        # Multiprocessing here
        with ProcessPoolExecutor(max_workers=workers) as executor:
            all_results = list(executor.map(sieve_at_offset, args_list))


        for rels in all_results:
            for rel in rels:
                if len(collected) >= max_relations:
                    break
                collected.append(rel)

        print(f"    Progress: collected {len(collected)} relations")
        offset += offset_step * 3

    return collected


# ============================================================================
# Linear Algebra Functions
# ============================================================================

def _pack_row_bits(args: Tuple) -> Tuple[int, np.ndarray]:
    """Helper function to pack matrix row into bit array."""
    idx, rel, factor_base = args
    cols = len(factor_base)
    words = (cols + 63) // 64
    packed = np.zeros(words, dtype=np.uint64)
    
    for j, p in enumerate(factor_base):
        exp = rel.factors.get(p, 0) if p != -1 else rel.factors.get(-1, 0)
        bit = exp & 1
        if bit:
            word_index = j // 64
            bit_index = j % 64
            packed[word_index] |= (1 << bit_index)
    
    return idx, packed


def create_matrix_bitpacked(
    relations: List[Relation], 
    factor_base: List[int], 
    workers: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Create bit-packed matrix from relations.
    
    Args:
        relations: List of smooth relations
        factor_base: Factor base primes
        workers: Number of worker processes
        
    Returns:
        Bit-packed matrix and number of columns
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)
    
    num_rows = len(relations)
    num_cols = len(factor_base)
    
    args = [(i, relations[i], factor_base) for i in range(num_rows)]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_pack_row_bits, args))
    
    results.sort(key=lambda x: x[0])
    matrix = np.vstack([r[1] for r in results]).astype(np.uint64)
    
    return matrix, num_cols


def _unpack_bits_uint64_array(bitarr: np.ndarray, nbits: int) -> List[int]:
    """Unpack bit array to list of integers."""
    res = [0] * nbits
    for word_idx, word in enumerate(bitarr):
        base = word_idx * 64
        w = int(word)
        if w == 0:
            continue
        for b in range(64):
            i = base + b
            if i >= nbits:
                break
            if (w >> b) & 1:
                res[i] = 1
    return res


def bitpacked_gaussian_nullspace(
    matrix_uint64: np.ndarray, 
    num_cols: int
) -> List[List[int]]:
    """
    Find nullspace of bit-packed matrix using Gaussian elimination.
    
    Args:
        matrix_uint64: Bit-packed matrix
        num_cols: Number of columns
        
    Returns:
        List of dependency vectors
    """
    if matrix_uint64.dtype != np.uint64:
        raise ValueError("Matrix must be dtype uint64")
    
    num_rows = matrix_uint64.shape[0]
    words_cols = matrix_uint64.shape[1]
    words_rows = (num_rows + 63) // 64
    
    # Create augmented matrix
    A = matrix_uint64.copy()
    Aug = np.zeros((num_rows, words_rows), dtype=np.uint64)
    for r in range(num_rows):
        w = r // 64
        b = r % 64
        Aug[r, w] = (1 << b)
    
    # Gaussian elimination
    rank = 0
    for col in range(num_cols):
        word_idx = col // 64
        bit_idx = col % 64
        mask = (np.uint64(1) << np.uint64(bit_idx))
        
        # Find pivot
        pivot = -1
        for r in range(rank, num_rows):
            if (A[r, word_idx] & mask) != 0:
                pivot = r
                break
        
        if pivot == -1:
            continue
        
        # Swap rows if needed
        if pivot != rank:
            A[[pivot, rank]] = A[[rank, pivot]]
            Aug[[pivot, rank]] = Aug[[rank, pivot]]
        
        # Eliminate
        pivot_row_A = A[rank].copy()
        pivot_row_Aug = Aug[rank].copy()
        
        for r in range(num_rows):
            if r == rank:
                continue
            if (A[r, word_idx] & mask) != 0:
                A[r] ^= pivot_row_A
                Aug[r] ^= pivot_row_Aug
        
        rank += 1
        if rank >= num_rows:
            break
    
    # Extract dependencies
    dependencies = []
    zero_row_mask = np.zeros(words_cols, dtype=np.uint64)
    for r in range(num_rows):
        if np.array_equal(A[r], zero_row_mask):
            dependency_vector = _unpack_bits_uint64_array(Aug[r], num_rows)
            if any(dependency_vector):
                dependencies.append(dependency_vector)
    
    return dependencies

# ============================================================================
# Factor Extraction
# ============================================================================

def find_factor_from_dependency(
    N: int, 
    relations: List[Relation], 
    dependency: List[int], 
    factor_base: List[int]
) -> Optional[int]:
    """
    Attempt to find a factor of N using a linear dependency.
    
    Args:
        N: Number to factor
        relations: List of smooth relations
        dependency: Dependency vector
        factor_base: Factor base primes
        
    Returns:
        A non-trivial factor of N, or None
    """
    sqrtN = isqrt(N)
    x_product = 1
    combined_factors = defaultdict(int)
    
    # Combine relations according to dependency
    for i, use_relation in enumerate(dependency):
        if use_relation == 1:
            rel = relations[i]
            x_val = sqrtN + rel.x
            x_product = (x_product * x_val) % N
            
            for p, exp in rel.factors.items():
                combined_factors[p] += exp
    
    # Check that all exponents are even
    for p, total_exp in combined_factors.items():
        if total_exp % 2 != 0:
            return None
    
    # Compute y as product of square roots
    y = 1
    for p, total_exp in combined_factors.items():
        if p != -1:
            y = (y * pow(p, total_exp // 2, N)) % N
    
    if combined_factors.get(-1, 0) % 2 == 1:
        y = (-y) % N
    
    # Check for trivial congruence
    if x_product == y or x_product == (-y) % N:
        return None
    
    # Try to find factor
    gcd1 = math.gcd(x_product - y, N)
    if 1 < gcd1 < N:
        return gcd1
    
    gcd2 = math.gcd(x_product + y, N)
    if 1 < gcd2 < N:
        return gcd2
    
    return None

# ============================================================================
# Main Quadratic Sieve Algorithm
# ============================================================================

def select_parameters(bit_size: int) -> Tuple[int, int]:
    """
    Select optimal parameters based on bit size of N.
    
    Args:
        bit_size: Number of bits in N
        
    Returns:
        Tuple of (factor_base_bound, sieve_range)
    """
    params = {
        30: (800, 1500),
        40: (1000, 3000),
        50: (5000, 14000),
        60: (10000, 18000),
        70: (30000, 60000),
        80: (60000, 120000),
        90: (150000, 300000),
        100: (300000, 450000),
        110: (1100000, 1700000),
        120: (1200000, 1800000),
        128: (1800000, 3200000),
    }
    
    for size, (B, M) in params.items():
        if bit_size <= size:
            return B, M
    
    # Default for larger numbers
    B = min(30000, bit_size * 250)
    M = min(100000, bit_size * 800)
    return B, M

# =====================================================================
# System Monitoring
# =====================================================================

class SystemMonitor(threading.Thread):
    """Tracks average CPU load and peak CPU clock speed."""
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
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

    @property
    def average_cpu(self):
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0


def system_metrics_before():
    """Capture initial system metrics."""
    battery = psutil.sensors_battery()
    return {
        "memory": psutil.virtual_memory().used / (1024 * 1024),
        "battery": battery.percent if battery else None,
        "charging": battery.power_plugged if battery else None,
    }

def system_metrics_after(before):
    """Capture system metrics after benchmark."""
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


def quadratic_sieve(N: int) -> Optional[Tuple[int, int]]:
    """
    Factor N using the Quadratic Sieve algorithm.
    
    Args:
        N: Integer to factor
        
    Returns:
        Tuple (p, q) where N = p * q, or None if factorization fails
    """
    bit_size = N.bit_length()
    print(f"Factoring {N} ({bit_size} bits)")
    
    # Check small primes first
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if N % p == 0:
            q = N // p
            return p, q
    
    # Select parameters
    B, M = select_parameters(bit_size)
    print(f"    Using factor base bound B = {B}, sieve range M = {M}")
    
    # Generate factor base
    factor_base = factor_base_primes(N, B)
    print(f"    Factor base size: {len(factor_base)}")
    
    # Precompute logarithms
    log_p_cache = {p: math.log(p) for p in factor_base if p > 0}
    
    # Calculate target relations
    overshoot = 1.4
    target_relations = int(len(factor_base) * overshoot) + 20
    print(f"    Target relations: {target_relations}")
    
    # Collect smooth relations
    relations = optimized_sieve_incremental(
        N, M, factor_base, log_p_cache, 
        max_relations=target_relations
    )
    print(f"    Found {len(relations)} smooth relations")
    
    # Check if we have enough relations
    if len(relations) < len(factor_base) + 10:
        print(f"    Not enough relations. Need ~{len(factor_base) + 10}")
        return None
    
    # Create matrix
    print(f"    Creating {len(relations)}×{len(factor_base)} matrix")
    matrix_bitpacked, num_cols = create_matrix_bitpacked(relations, factor_base)
    
    # Find dependencies
    dependencies = bitpacked_gaussian_nullspace(matrix_bitpacked, num_cols)
    print(f"    Found {len(dependencies)} dependencies")
    
    if not dependencies:
        print("    No dependencies found")
        return None
    
    # Try each dependency
    for i, dep in enumerate(dependencies):
        print(f"    Trying dependency {i+1}/{len(dependencies)}")
        factor = find_factor_from_dependency(N, relations, dep, factor_base)
        if factor:
            p = factor
            q = N // factor
            if p > q:
                p, q = q, p
            print(f"    Found factor using dependency {i+1}")
            return p, q
    
    print("    All dependencies failed to produce factors")
    return None

# =====================================================================
# Batch Runner
# =====================================================================

def run_batch(semiprimes: List[int]):
    """Runs QS on multiple semiprimes and reports overall metrics."""
    print("=" * 70)
    print(f"Starting Quadratic Sieve Batch Benchmark ({len(semiprimes)} semiprimes)")
    print("=" * 70)

    before = system_metrics_before()
    if before["charging"]:
        print("⚠️  Device is charging. Skipping benchmark to avoid interference.")
        return

    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    start_time = timeit.default_timer()
    successes = 0

    for n in semiprimes:
        print(f"\nFactoring {n}...")
        result = quadratic_sieve(n)
        if result and result[0] * result[1] == n:
            print(f"✓ Success: {n} = {result[0]} × {result[1]}")
            successes += 1
        else:
            print(f"✗ Failed on {n}")

    total_runtime = timeit.default_timer() - start_time

    monitor.stop()
    monitor.join()

    after = system_metrics_after(before)

    print("\n" + "=" * 70)
    print("Batch Summary")
    print("=" * 70)
    print(f"Total Semiprimes Tested: {len(semiprimes)}")
    print(f"Successful Runs: {successes}/{len(semiprimes)}")
    print(f"Success Rate: {(successes / len(semiprimes)) * 100:.2f}%")
    print("\n--- Performance Metrics ---")
    print(f"Total Runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"Throughput: {successes / total_runtime:.4f} successful factorizations/second")
    print(f"Average CPU Load: {monitor.average_cpu:.2f}%")
    print(f"Peak RAM Usage: {after['peak_memory']:.2f} MB")
    print(f"Peak CPU Clock Speed: {monitor.peak_freq:.2f} MHz")
    if after["battery_before"] is not None:
        print(f"Battery Before: {after['battery_before']}%")
        print(f"Battery After: {after['battery_after']}%")
        print(f"Battery Consumption: {after['battery_consumption']:.2f}%")
    print("=" * 70)

# =====================================================================
# Main Entry
# =====================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algorithm_name = "QS_Batch"
    save_dir = r"C:\Users\Frances Bea Magdayao\Desktop\Thesisenv\Objective 2\QS AW logs"
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"{algorithm_name}_{timestamp}.txt")

    sys.stdout = DualLogger(log_filename)
    print(f"Log started at {timestamp}")
    print(f"Output file: {log_filename}")
    print("=" * 70)

    semiprimes = [
        207073211425765647323701979,
        2405668863350329627101536027,
        413382272494408950787512431
    ]

    run_batch(semiprimes)

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✅ Output also saved to:\n{log_filename}")
