"""Technically this is not pairwise yet since it only computes the scalar distance between
two vectors. To expand to pairwise such that we can take in 2 matrices and output pairwise
distances between the rows/columns of the matrices."""
import numpy as np
import scipy


def manhattan_distance(x_1: np.ndarray, x_2: np.ndarray) -> float:
    """
    Calculates the Manhattan Distance between two data points.

    The Manhattan Distance is the sum of the absolute differences of two vectors' Cartesian coordinates.
    For two vectors a, b, it can be defined as:

        L1(a, b) = |a_1-b_1| + |a_2-b_2| + ... + |a_n-b_n|

    The Manhattan Distance is also referred to as the L1 distance or L1 vector norm. It generalizes to N-dimensional
    Euclidean Space. For example, consider 2 points in 3D (x1,y1, z1), (x2, y2, z2) then the manhattan distance
    between them is given by |x1-x2| + |y1-y2| + |z1-z2|.

    Args:
        x_1 (np.ndarray): A numpy array representing the first data point.
        x_2 (np.ndarray): A numpy array representing the second data point.

    Returns:
        _manhattan_distance (float): The Manhattan distance between the two data points.

    Example:
        >>> x_1 = [1, 2, 3]
        >>> x_2 = [2, 3, 5]
        >>> manhattan_distance(x_1, x_2)
        4
    """
    _manhattan_distance = np.sum(np.abs(x_1 - x_2))
    return _manhattan_distance


def euclidean_distance(
    x_1: np.ndarray, x_2: np.ndarray, squared: bool = False
) -> float:
    """
    Euclidean Distance measures the length of the line segment bewteen two points
    in the Euclidean space.  This is also referred to as L2 distance or L2 vector norm.


        L2(a, b) = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2 + \cdots + (a_n-b_n)^2}


    Generalizes to N-dimensional Euclidean Space:
        Consider 2 points in 3D (x1,y1, z1), (x2, y2, z2) then the
        euclidean distance between them is given by
        \sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
        In code, one can easily use summation(x_true-x_new)^2 because we can reduce above formula
        to \sum_{i=1}^N vector_1[i] - vector_2[i] ^2

    Args:
        x_1 (np.ndarray): a data point in numpy array of dimension D
        x_2 (np.ndarray): a data point in numpy array of dimension D

    Returns:
        _euclidean_distance (float): the Euclidean distance between the two data points
    """
    if not squared:
        _euclidean_distance = np.sum(np.square(x_1 - x_2))
    else:
        _euclidean_distance = np.sqrt(np.sum(np.square(x_1 - x_2)))
    return _euclidean_distance


def cosine_similarity(x_1: np.ndarray, x_2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        x_1 (np.ndarray): A numpy array representing the first vector.
        x_2 (np.ndarray): A numpy array representing the second vector.

    Returns:
        _cosine_similarity (float): A float value representing the cosine similarity between the two vectors.

    Raises:
        ValueError: If either of the input vectors is empty.
        AssertionError: If the calculated norm of x_1 is not equal to the result of calculating the Euclidean
                        distance between x_1 and the origin.
    """

    numerator = np.dot(x_1, x_2)
    origin = np.zeros(shape=(x_1.shape))  # origin is a vector of zeros
    norm_x1 = np.linalg.norm(x_1)
    norm_x2 = np.linalg.norm(x_2)

    np.testing.assert_allclose(norm_x1, euclidean_distance(x_1, origin, squared=True))

    denominator = norm_x1 * norm_x2
    _cosine_similarity = numerator / denominator
    return _cosine_similarity


def cosine_distance(x_1: np.ndarray, x_2: np.ndarray) -> float:
    return 1 - cosine_similarity(x_1, x_2)


if __name__ == "__main__":
    ### Euclidean Distance Testing ###
    test_distance = np.array([[-4, -3, -2], [-1, 0, 1]])

    print(np.linalg.norm(test_distance[0] - test_distance[1]))
    print(np.linalg.norm(test_distance[0] - test_distance[1]).reshape(-1, 1))
    print(euclidean_distance(test_distance[0], test_distance[1]))
    print(scipy.spatial.distance.euclidean(test_distance[0], test_distance[1]))

    print(manhattan_distance(test_distance[0], test_distance[1]))
    print(scipy.spatial.distance.minkowski(test_distance[0], test_distance[1], p=1))

    print(scipy.spatial.distance.cosine(test_distance[0], test_distance[1], w=None))
    print(cosine_distance(test_distance[0], test_distance[1]))

    ### CS3244 ###

    x1 = np.asarray([1, 2, 3])
    x2 = np.asarray([0, 0, 0])
    print(
        "Euclidean Distance between (1, 2, 3) and (0, 0, 0):",
        euclidean_distance(x1, x2),
    )
    print(
        "Manhattan Distance between (1, 2, 3) and (0, 0, 0):",
        manhattan_distance(x1, x2),
    )

    x1 = np.asarray([100, 20, 30])
    x2 = np.asarray([0, 0, 0])
    print(
        "Euclidean Distance between (100, 20, 30) and (0, 0, 0):",
        euclidean_distance(x1, x2),
    )
    print(
        "Manhattan Distance between (100, 20, 30) and (0, 0, 0):",
        manhattan_distance(x1, x2),
    )
