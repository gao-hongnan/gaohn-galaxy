import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from src.neighbours.knn.knn import KNNClassifier, KNNRegressor
from src.fundamentals.decision_boundary.decision_boundary import plot_decision_regions


def classification():
    """Classification for KNN"""

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.7, random_state=42
    )

    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    sklearn_predictions = sklearn_knn.fit(X_train, y_train).score(X_test, y_test)

    HN_KNN_CLASSIFICATION = KNNClassifier(num_neighbours=3, metric="euclidean")
    HN_KNN_CLASSIFICATION.fit(X_train, y_train)
    HN_CLASSIFICATION_PREDICTIONS = HN_KNN_CLASSIFICATION.predict(
        X_train, y_train, X_test
    )

    print(HN_CLASSIFICATION_PREDICTIONS)
    print("\nSKLEARN Accuracy score : %.3f" % (sklearn_predictions * 100))
    print(
        "\nHN Accuracy score : %.3f"
        % (accuracy_score(y_test, HN_CLASSIFICATION_PREDICTIONS) * 100)
    )
    print()

    # print(sklearn_knn.fit(X_train, y_train).predict_proba(X_test))
    # print("Recall score : %f" % (sklearn.metrics.recall_score(y_val, preds) * 100))
    # print("ROC score : %f\n" % (sklearn.metrics.roc_auc_score(y_val, preds) * 100))
    # print(sklearn.metrics.confusion_matrix(y_val, preds))


def regression():
    """Regression for KNN"""
    X = np.array([[0], [1], [2], [3]])

    y = np.array([0, 0, 1, 1])

    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X, y)

    print(neigh.predict([[1.5]]))
    HN_KNN_REGRESSION = KNNRegressor(num_neighbours=2, metric="euclidean")
    HN_KNN_REGRESSION.fit(X, y)
    HN_REGRESSION_PREDICTIONS = HN_KNN_REGRESSION.predict(X, y, np.array([[1.5]]))

    print(HN_REGRESSION_PREDICTIONS)


def plot_knn_classifier_decision_boundary():
    """Plot Decision Boundary of KNN"""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.7, random_state=42
    )
    X_train_2d = X_train[:, :2]
    # knn_classifier = KNNClassifier(num_neighbours=5, metric="euclidean")
    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(X_train_2d, y_train)
    plot_decision_regions(
        X_train_2d,
        y_train,
        classifier=sklearn_knn,
        contourf=True,
        alpha=0.3,
    )


if __name__ == "__main__":
    classification()
    regression()
    plot_knn_classifier_decision_boundary()
