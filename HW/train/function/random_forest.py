"""random_forest.py

实现一个花哨的随机森林分类器，结合 scikit-learn 和自定义可视化。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report


def generate_data():
    """生成用于分类的随机数据集。"""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    return X, y


def train_random_forest(X, y):
    # function random_forest.py is an
    """训练随机森林分类器。"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    return model


def visualize_trees(model, feature_names):
    """可视化随机森林中的部分决策树。"""
    for i in range(min(3, len(model.estimators_))):
        plt.figure(figsize=(20, 10))
        plot_tree(
            model.estimators_[i],
            feature_names=feature_names,
            filled=True,
            rounded=True,
            class_names=[str(cls) for cls in np.unique(model.classes_)]
        )
        plt.title(f"Decision Tree {i+1}")
        plt.show()


def evaluate_model(model, X, y):
    """评估模型性能。"""
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred))


if __name__ == "__main__":
    # 生成数据
    X, y = generate_data()
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # 训练随机森林
    model = train_random_forest(X, y)

    # 可视化部分决策树
    visualize_trees(model, feature_names)

    # 评估模型
    evaluate_model(model, X, y)