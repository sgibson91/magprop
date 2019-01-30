import numpy as np


def geomean(list_a, list_b):
    """
Function to calculate the geometric mean of pairs of numbers.

    :param list_a: list or array of positive uncertainties
    :param list_b: list or array of negative uncertainties
    :return: list or array of the geometric mean of the pairs of numbers between
             list_a and list_b
    """
    return [np.prod([i, np.abs(j)]) ** 0.5 for i, j in zip(list_a, list_b)]


def test_geomean():
    list_a = [1.0, 2.0, 3.0]
    list_b = [4.0, 5.0, 6.0]

    ans = []
    for x, y in zip(list_a, list_b):
        ans.append((x * y) ** 0.5)

    assert geomean(list_a, list_b) == ans

