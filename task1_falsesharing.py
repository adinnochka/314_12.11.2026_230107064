"""Phase 1: shared-memory false-sharing experiment; stdlib only.

Default follows lab protocol: 24m total updates, 2 warmups + 7 trials.
Use --quick for a preliminary functionality check, not final measurements.
"""
import argparse
import csv
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230107064
MASK = (1 << 64) - 1
MULT = 6364136223846793005


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = None
    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
        acc = arr[slot_idx]
        for i in range(iters):
            acc = (acc * MULT + (i | 1)) & MASK
            arr[slot_idx] = acc
    finally:
        del arr
        shm.close()


def trial(p, stride, total_ops):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        counts = [total_ops // p + (1 if t < total_ops % p else 0) for t in range(p)]
        workers = [mp.Process(target=fs_worker, args=(shm.name, t * stride, counts[t])) for t in range(p)]
        for proc in workers:
            proc.start()
        for proc in workers:
            proc.join()
        if any(proc.exitcode != 0 for proc in workers):
            raise RuntimeError(f'Worker failed: {[proc.exitcode for proc in workers]}')
    finally:
        shm.close()
        shm.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-ops', type=int, default=24_000_000)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--physical-cores', type=int, default=8)
    parser.add_argument('--logical-cores', type=int, default=16)
    parser.add_argument('--quick', action='store_true', help='Smoke test only, NOT submission data')
    args = parser.parse_args()
    if args.quick:
        args.total_ops, args.warmups, args.runs = 20_000, 0, 1
    if args.total_ops < 1 or args.physical_cores < 1 or args.logical_cores < 1:
        parser.error('Counts must be positive')
    if args.logical_cores > 16:
        parser.error('This lab sweep supports up to 16 logical workers for 4096-byte buffer')
    print('PHASE 1 — FALSE SHARING')
    print(f'ID={STUDENT_ID} M={args.total_ops} warmups={args.warmups} timed_runs={args.runs}')
    if args.quick:
        print('QUICK TEST ONLY: rerun without --quick for valid lab data')
    rows = []

    def measure(kind, p, stride):
        stats = run_benchmark(lambda: trial(p, stride, args.total_ops), args.warmups, args.runs)
        row = {'experiment': kind, 'p': p, 'stride': stride, 'offset_bytes': stride * 8,
               'median_ms': stats.median_ms, 'iqr_ms': stats.iqr_ms,
               'min_ms': stats.min_ms, 'max_ms': stats.max_ms,
               'throughput_mops': args.total_ops / (stats.median_ms * 1000),
               'speedup': '', 'efficiency': '', 'karp_flatt_e': ''}
        rows.append(row)
        print(f'{kind:7} p={p:2} stride={stride:2} median={stats.median_ms:.4f}ms IQR={stats.iqr_ms:.4f}ms')
        return row

    for stride in (1, 2, 4, 8, 16, 32):
        measure('stride', args.physical_cores, stride)

    for stride in (1, 16):
        ps = sorted(set([1, 2, 4, 8, 16, args.logical_cores]))
        ps = [p for p in ps if p <= args.logical_cores]
        series = [measure('scaling', p, stride) for p in ps]
        baseline = series[0]['median_ms']
        for row in series:
            p = row['p']
            speedup = baseline / row['median_ms']
            row['speedup'] = speedup
            row['efficiency'] = speedup / p
            row['karp_flatt_e'] = ((1 / speedup - 1 / p) / (1 - 1 / p)) if p > 1 else 0.0

    path = Path(__file__).resolve().parent / ('phase1_quick.csv' if args.quick else 'phase1_results.csv')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    s1 = next(r for r in rows if r['experiment'] == 'stride' and r['stride'] == 1)
    s16 = next(r for r in rows if r['experiment'] == 'stride' and r['stride'] == 16)
    print('Saved:', path)
    print(emit_verify_tag('PHASE1', STUDENT_ID, s1['median_ms'], s16['median_ms']))
    if args.quick:
        print('Do not use this quick-test hash in the final report.')


if __name__ == '__main__':
    mp.freeze_support()
    main()
