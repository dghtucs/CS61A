import random
import math
import time
from typing import List, Sequence


def insertion_sort(a: List, left: int, right: int) -> None:
    for i in range(left + 1, right + 1):
        key = a[i]
        j = i - 1
        while j >= left and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


def heapify(a: List, n: int, i: int) -> None:
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and a[l] > a[largest]:
        largest = l
    if r < n and a[r] > a[largest]:
        largest = r
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        heapify(a, n, largest)


def heapsort(a: List, left: int, right: int) -> None:
    # Build heap on slice a[left:right+1]
    n = right - left + 1
    if n <= 1:
        return
    # shift to 0-based for heap operations
    temp = a[left:right + 1]
    for i in range(n // 2 - 1, -1, -1):
        heapify(temp, n, i)
    for i in range(n - 1, 0, -1):
        temp[0], temp[i] = temp[i], temp[0]
        heapify(temp, i, 0)
    a[left:right + 1] = temp


def median_of_three(a: List, i: int, j: int, k: int):
    A = a[i]
    B = a[j]
    C = a[k]
    if A < B:
        if B < C:
            return j
        return k if A < C else i
    else:
        if A < C:
            return i
        return k if B < C else j


def _introsort(a: List, left: int, right: int, maxdepth: int) -> None:
    while left < right:
        size = right - left + 1
        if size <= 16:
            insertion_sort(a, left, right)
            return
        if maxdepth == 0:
            heapsort(a, left, right)
            return
        # pivot = median-of-three
        m = median_of_three(a, left, left + size // 2, right)
        a[m], a[right] = a[right], a[m]
        pivot = a[right]
        i = left
        for j in range(left, right):
            if a[j] < pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[right] = a[right], a[i]
        # recurse to smaller partition first (tail recursion elimination)
        left_size = i - left
        right_size = right - i
        if left_size < right_size:
            _introsort(a, left, i - 1, maxdepth - 1)
            left = i + 1
        else:
            _introsort(a, i + 1, right, maxdepth - 1)
            right = i - 1


def introsort(a: List) -> None:
    if not a:
        return
    maxdepth = (math.floor(math.log2(len(a))) + 1) * 2
    _introsort(a, 0, len(a) - 1, maxdepth)


def radix_sort_int(a: List[int]) -> List[int]:
    # LSD radix sort for 32-bit signed integers
    if not a:
        return a
    # work with copy to avoid mutating user list
    b = list(a)
    # offset negatives by 2**31 to map into unsigned space, or handle sign later
    OFFSET = 1 << 31
    for i in range(len(b)):
        b[i] = (b[i] + OFFSET) & 0xFFFFFFFF
    MASK = 0xFF
    for shift in (0, 8, 16, 24):
        bins = [0] * 256
        for x in b:
            bins[(x >> shift) & MASK] += 1
        for i in range(1, 256):
            bins[i] += bins[i - 1]
        out = [0] * len(b)
        for x in reversed(b):
            idx = (x >> shift) & MASK
            bins[idx] -= 1
            out[bins[idx]] = x
        b = out
    # map back
    for i in range(len(b)):
        b[i] = (b[i] - OFFSET) & 0xFFFFFFFF
        # convert to signed
        if b[i] & (1 << 31):
            b[i] = b[i] - (1 << 32)
    return b


def sort_fast(a: Sequence) -> List:
    # Dispatcher: use radix for ints, otherwise introsort in-place
    if not a:
        return []
    if all(isinstance(x, int) for x in a):
        return radix_sort_int(list(a))
    else:
        arr = list(a)
        introsort(arr)
        return arr


def mergesort(a: Sequence) -> List:
    a = list(a)
    n = len(a)
    if n <= 1:
        return a
    mid = n // 2
    left = mergesort(a[:mid])
    right = mergesort(a[mid:])
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def pure_quicksort(a: Sequence) -> List:
    a = list(a)
    if len(a) <= 1:
        return a
    pivot = a[len(a) // 2]
    left = [x for x in a if x < pivot]
    mid = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return pure_quicksort(left) + mid + pure_quicksort(right)


def _time(func, data, repeat=3):
    times = []
    for _ in range(repeat):
        arr = list(data)
        t0 = time.perf_counter()
        func(arr)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), sum(times) / len(times)


def _bench_once(n=100000, typ='int'):
    if typ == 'int':
        data = [random.randint(-10**9, 10**9) for _ in range(n)]
    elif typ == 'float':
        data = [random.random() * 1e9 for _ in range(n)]
    else:
        data = [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8)) for _ in range(n)]

    funcs = [
        ('built-in', lambda x: x.sort()),
        ('sorted', lambda x: sorted(x)),
        ('introsort', lambda x: introsort(x)),
        ('radix_or_fast', lambda x: sort_fast(x)),
        ('mergesort', lambda x: mergesort(x)),
        ('py_quick', lambda x: pure_quicksort(x)),
    ]

    results = {}
    for name, f in funcs:
        tmin, tav = _time(f, data, repeat=3)
        results[name] = (tmin, tav)
    return results


def run_quick_bench():
    print('Quick benchmark (smaller sizes to keep runtime short)')
    for n in (1000, 5000, 20000):
        print(f'\nN = {n}')
        r = _bench_once(n=n, typ='int')
        for k, v in r.items():
            print(f'  {k:12}: min={v[0]:.6f}s avg={v[1]:.6f}s')


if __name__ == '__main__':
    run_quick_bench()
