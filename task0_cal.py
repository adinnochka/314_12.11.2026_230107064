import multiprocessing as mp
import platform
import sys
import time
from harness import emit_verify_tag

STUDENT_ID = 230107064
READS = 2_000_000

def main():
    print('PHASE 0 — HARDWARE AND TIMER CALIBRATION')
    print('Student ID:', STUDENT_ID)
    print('Platform:', platform.platform())
    print('Processor:', platform.processor())
    print('Python:', sys.version.split()[0])
    print('Logical CPUs:', mp.cpu_count())
    print('Timer reads:', READS)
    t0 = time.perf_counter_ns()
    for _ in range(READS):
        time.perf_counter_ns()
    elapsed_ns = time.perf_counter_ns() - t0
    avg_ns = elapsed_ns / READS
    print(f'Total loop time: {elapsed_ns / 1e9:.6f} s')
    print(f'Average timer call + loop overhead: {avg_ns:.4f} ns')
    print('NOTE: Includes Python loop overhead; not pure hardware timer latency.')
    print(emit_verify_tag('PHASE0', STUDENT_ID, float(mp.cpu_count()), avg_ns))

if __name__ == '__main__':
    main()
