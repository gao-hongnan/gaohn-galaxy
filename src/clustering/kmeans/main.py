"""Main script to run the K-Means algorithm."""
from __future__ import annotations

import numpy as np
from rich.pretty import pprint
from rich import print
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split

from src.clustering.kmeans.kmeans import KMeansLloyd
from src.metrics.clustering.supervised import contingency_matrix, purity_score

# from sklearn.metrics.cluster import contingency_matrix

if __name__ == "__main__":
    # sklearn example
    X = np.array(
        [
            [1, 2],
            [1, 4],
            [1, 0],
            [10, 2],
            [10, 4],
            [10, 0],
        ]
    )
    kmeans = KMeansLloyd(num_clusters=2, init="random", max_iter=500, random_state=1992)
    kmeans.fit(X)
    pprint(f"There are {kmeans.num_clusters} clusters.")
    print(f"The centroids are\n{kmeans.centroids}.")
    pprint(f"The labels predicted are {kmeans.labels}.")
    pprint(f"The inertia is {kmeans.inertia}.")
    pprint(f"The clusters are {kmeans.clusters}.")

    y_preds = kmeans.predict([[0, 0], [12, 3]])
    pprint(f"The predicted labels for new data are {y_preds}.")

    sk_kmeans = KMeans(
        n_clusters=2, random_state=1992, n_init="auto", algorithm="lloyd", max_iter=500
    )
    sk_kmeans.fit(X)
    print(f"The centroids are\n{sk_kmeans.cluster_centers_}.")
    pprint(f"The labels predicted are {sk_kmeans.labels_}.")
    pprint(f"The inertia is {sk_kmeans.inertia_}.")

    y_preds = sk_kmeans.predict([[0, 0], [12, 3]])
    pprint(f"The predicted labels for new data are {y_preds}.")

    ####### IRIS DATASET #######
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    kmeans = KMeansLloyd(num_clusters=3, init="random", max_iter=500, random_state=2023)
    kmeans.fit(X_train)
    pprint(kmeans.labels)
    pprint(kmeans.inertia)

    y_preds = kmeans.predict(X_train)
    assert np.all(y_preds == kmeans.labels)
    contingency_matrix_ = contingency_matrix(y_train, y_preds)
    pprint(contingency_matrix_)
    # TODO: try K = 4

    purity = purity_score(y_train, y_preds)
    pprint(purity)
    purity_per_cluster = purity_score(y_train, y_preds, per_cluster=True)
    pprint(purity_per_cluster)

    sk_kmeans = KMeans(
        n_clusters=3,
        random_state=2023,
        n_init=1,
        algorithm="lloyd",
        max_iter=500,
        init="random",
    )
    sk_kmeans.fit(X_train)
    pprint(sk_kmeans.labels_)
    pprint(sk_kmeans.inertia_)

    y_preds = sk_kmeans.predict(X_train)
    assert np.all(y_preds == sk_kmeans.labels_)
    contingency_matrix_ = contingency_matrix(y_train, y_preds)
    pprint(contingency_matrix_)

    purity = purity_score(y_train, y_preds)
    pprint(purity)

    ####### MNIST DATASET #######
    X, y = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = X.shape, np.unique(y).size

    pprint(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    kmeans = KMeansLloyd(
        num_clusters=n_digits, init="random", max_iter=50000, random_state=42
    )
    kmeans.fit(X_train)
    pprint(kmeans.labels)
    pprint(kmeans.inertia)

    y_preds = kmeans.predict(X_test)
    contingency_matrix_ = contingency_matrix(y_test, y_preds, as_dataframe=True)
    pprint(contingency_matrix_)
    purity = purity_score(y_test, y_preds)
    pprint(purity)
    purity_per_cluster = purity_score(y_test, y_preds, per_cluster=True)
    pprint(purity_per_cluster)

    sk_kmeans = KMeans(
        n_clusters=n_digits,
        random_state=42,
        n_init=1,
        max_iter=500,
        init="random",
        algorithm="lloyd",
    )
    sk_kmeans.fit(X_train)
    # Note that the labels can be permuted.
    pprint(sk_kmeans.labels_)
    pprint(sk_kmeans.inertia_)
