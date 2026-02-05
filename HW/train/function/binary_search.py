"""binary_search.py

提供迭代与递归两种二分查找实现。
"""
from typing import List


def binary_search_iterative(arr: List[int], target: int) -> int:
    """在有序数组 arr 中查找 target，返回索引或 -1（未找到）。

    时间复杂度：O(log n)
    要求：arr 必须是已排序（升序）的序列。
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def binary_search_recursive(arr: List[int], target: int, lo: int = 0, hi: int = None) -> int:
    """递归实现的二分查找，返回索引或 -1（未找到）。"""
    if hi is None:
        hi = len(arr) - 1
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, hi)
    else:
        return binary_search_recursive(arr, target, lo, mid - 1)


def binary_search_pythonic(arr: List[int], target: int) -> int:
    """使用 Pythonic 风格实现的二分查找，返回索引或 -1（未找到）。"""
    from bisect import bisect_left

    idx = bisect_left(arr, target)
    if idx != len(arr) and arr[idx] == target:
        return idx
    return -1


if __name__ == "__main__":
    sample = [1, 3, 5, 7, 9, 11]
    print("iterative search for 7 ->", binary_search_iterative(sample, 7))
    print("recursive search for 2 ->", binary_search_recursive(sample, 2))
    print("pythonic search for 9 ->", binary_search_pythonic(sample, 9))
