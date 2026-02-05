"""test_random_forest.py

测试随机森林实现的功能。
"""
import unittest
from random_forest import generate_data, train_random_forest, evaluate_model


class TestRandomForest(unittest.TestCase):
    def test_data_generation(self):
        X, y = generate_data()
        self.assertEqual(X.shape[0], 500)
        self.assertEqual(y.shape[0], 500)
        self.assertEqual(X.shape[1], 10)

    def test_model_training(self):
        X, y = generate_data()
        model = train_random_forest(X, y)
        self.assertEqual(len(model.estimators_), 100)
        self.assertEqual(model.n_classes_, 3)

    def test_model_evaluation(self):
        X, y = generate_data()
        model = train_random_forest(X, y)
        try:
            evaluate_model(model, X, y)
        except Exception as e:
            self.fail(f"evaluate_model raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()