def bubbleSort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(i+1,n):
            if(arr[i] > arr[j]):
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp
    return arr


def bubble_sort(arr):
    """标准冒泡排序（相邻比较），带提前退出优化。"""
    n = len(arr)
    # 复制一份以避免就地修改调用者传入的列表
    a = list(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


if __name__ == '__main__':
    samples = [[], [1], [2,1], [3,1,2], [5,3,4,1,2]]
    for s in samples:
        print('input:', s)
        print('bubbleSort ->', bubbleSort(s.copy()))
        print('bubble_sort ->', bubble_sort(s))




