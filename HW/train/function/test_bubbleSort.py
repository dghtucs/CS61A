import unittest
from bubbleSort import bubbleSort, bubble_sort


class TestBubbleSort(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(bubbleSort([]), [])
        self.assertEqual(bubble_sort([]), [])

    def test_single(self):
        self.assertEqual(bubbleSort([1]), [1])
        self.assertEqual(bubble_sort([1]), [1])

    def test_sorted(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(bubbleSort(arr.copy()), [1,2,3,4,5])
        self.assertEqual(bubble_sort(arr.copy()), [1,2,3,4,5])

    def test_reverse(self):
        arr = [5,4,3,2,1]
        self.assertEqual(bubbleSort(arr.copy()), [1,2,3,4,5])
        self.assertEqual(bubble_sort(arr.copy()), [1,2,3,4,5])

    def test_duplicates(self):
        arr = [3,1,2,3,1]
        self.assertEqual(bubbleSort(arr.copy()), sorted(arr))
        self.assertEqual(bubble_sort(arr.copy()), sorted(arr))

    def test_negative(self):
        arr = [0, -1, 5, -3]
        self.assertEqual(bubbleSort(arr.copy()), sorted(arr))
        self.assertEqual(bubble_sort(arr.copy()), sorted(arr))


if __name__ == '__main__':
    unittest.main()
