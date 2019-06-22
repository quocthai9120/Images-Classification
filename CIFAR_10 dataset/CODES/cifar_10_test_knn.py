# Student Name: Thai Quoc Hoang - uwnetid: qthai912
# Student ID: 1861609
# Section: CSE163 AC
# Instructor: Hunter Schafer

# Program Description: This program tests the functions the
# kNN approach with CIFAR_10 dataset.


import cifar_10_k_nearest_neighbors as knn
import numpy as np


def test_norm_k():
    '''
    Post: This function tests the correctness of norm_k method
    in the code file for kNN model (as imported). The function
    will return assertion error if the test fails.
    '''
    arr_1 = np.array([1, 2, 3])
    arr_2 = np.array([2, 3, 4])
    assert(1.732 - knn.norm_k(arr_1, arr_2) < 0.001)


def test_magnitude():
    '''
    Post: This function tests the correctness of magnitude method
    in the code file for kNN model (as imported). The function
    will return assertion error if the test fails.
    '''
    arr = np.array([1, 2, 3])
    assert(3.742 - knn.magnitude(arr) < 0.001)


def test_cosine_distance():
    '''
    Post: This function tests the correctness of cosine distance method
    in the code file for kNN model (as imported). The function
    will return assertion error if the test fails.
    '''
    arr_1 = np.array([1, 2, 3])
    arr_2 = np.array([2, 3, 4])
    assert(0.122 - knn.cosine_distance(arr_1, arr_2) < 0.001)


def test_get_k_nearest_neighbors():
    '''
    Post: This function tests the correctness of get_k_nearest_neighbors method
    in the code file for kNN model (as imported). The function
    will return assertion error if the test fails.
    '''
    x_train = np.arange(144).reshape((4, 4, 3, 3))
    y_train = np.array([1, 2, 3, 2])
    label_names = np.array(['zero', 'one', 'two', 'three'])
    x_test = np.arange(75, 111).reshape((4, 3, 3))
    assert(
        [('three', 18.0), ('two', 198.0), ('two', 234.0), ('one', 450.0)] ==
        knn.get_k_nearest_neighbors(x_train, y_train, label_names, x_test))


def test_get_result():
    '''
    Post: This function tests the correctness of get_result method
    in the code file for kNN model (as imported). The function
    will return assertion error if the test fails.
    '''
    x_train = np.arange(144).reshape((4, 4, 3, 3))
    y_train = np.array([1, 2, 3, 2])
    label_names = np.array(['zero', 'one', 'two', 'three'])
    x_test = np.arange(75, 111).reshape((4, 3, 3))

    assert('two' == knn.get_result(
            knn.get_k_nearest_neighbors(
                x_train, y_train, label_names, x_test)))
    assert([('two', 0.5), ('three', 0.25), ('one', 0.25)] == knn.get_result(
            knn.get_k_nearest_neighbors(
                x_train, y_train, label_names, x_test), verbose=1))


def main():
    print()
    print('Testing kNN methods')
    test_norm_k()
    test_magnitude()
    test_cosine_distance()
    test_get_k_nearest_neighbors()
    test_get_result()
    print('All tests passed')
    print()


if __name__ == "__main__":
    main()
