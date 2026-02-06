import unittest
from binary_search import binary_search_iterative, binary_search_recursive, binary_search_pythonic


class TestBinarySearch(unittest.TestCase):
    def test_found_beginning(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(binary_search_iterative(arr, 1), 0)
        self.assertEqual(binary_search_recursive(arr, 1), 0)
        self.assertEqual(binary_search_pythonic(arr, 1), 0)

    def test_found_middle(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(binary_search_iterative(arr, 3), 2)
        self.assertEqual(binary_search_recursive(arr, 3), 2)
        self.assertEqual(binary_search_pythonic(arr, 3), 2)

    def test_found_end(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(binary_search_iterative(arr, 5), 4)
        self.assertEqual(binary_search_recursive(arr, 5), 4)
        self.assertEqual(binary_search_pythonic(arr, 5), 4)

    def test_not_found(self):
        arr = [1, 2, 4, 6, 8]
        self.assertEqual(binary_search_iterative(arr, 3), -1)
        self.assertEqual(binary_search_recursive(arr, 3), -1)
        self.assertEqual(binary_search_pythonic(arr, 3), -1)

    def test_empty(self):
        arr = []
        self.assertEqual(binary_search_iterative(arr, 1), -1)
        self.assertEqual(binary_search_recursive(arr, 1), -1)
        self.assertEqual(binary_search_pythonic(arr, 1), -1)

    def test_single(self):
        arr = [7]
        self.assertEqual(binary_search_iterative(arr, 7), 0)
        self.assertEqual(binary_search_recursive(arr, 7), 0)
        self.assertEqual(binary_search_pythonic(arr, 7), 0)

    def test_duplicates(self):
        arr = [1, 2, 2, 2, 3]
        idx_it = binary_search_iterative(arr, 2)
        idx_rec = binary_search_recursive(arr, 2)
        idx_py = binary_search_pythonic(arr, 2)
        # 对重复元素，允许返回任何一个匹配的索引
        self.assertIn(idx_it, {1, 2, 3})
        self.assertIn(idx_rec, {1, 2, 3})
        self.assertIn(idx_py, {1, 2, 3})


if __name__ == "__main__":
    print("this is my fucntion")
    unittest.main()
