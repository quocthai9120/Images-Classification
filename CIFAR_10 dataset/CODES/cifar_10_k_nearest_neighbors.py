# Student Name: Thai Quoc Hoang - uwnetid: qthai912
# Student ID: 1861609
# Section: CSE163 AC
# Instructor: Hunter Schafer

# Program Description: This program does predicting and visualizing for
# the kNN approach with CIFAR_10 dataset.


import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns


def norm_k(np_arr1, np_arr2, k=2):
    '''
    Pre : gives 2 numpy arrays with equal length represents 2 vectors and
    an integer represents the norm value with default is 2.

    Post: return a float represents the norm of square distance
    for 2 given arrays.
    '''
    return np.sum((np_arr1 - np_arr2)**2) ** (1 / k)


def magnitude(np_arr):
    '''
    Pre : gives a numpy array of floats.

    Post: returns a float represents the magnitude of the array.
    '''
    return np.sum(np_arr**2)**0.5


def cosine_distance(np_arr1, np_arr2):
    '''
    Pre : gives 2 numpy arrays of float with equal length represents 2 vectors.

    Post: returns a float represents the cosine distance of two given vectors.
    '''
    return np.arccos(
        np.sum(np_arr1 * np_arr2) / (magnitude(np_arr1) * magnitude(np_arr2))
    )


def get_k_nearest_neighbors(x_train,
                            y_train,
                            label_names,
                            testing_instance,
                            k=5,
                            distance_function='norm_k',
                            norm=2
                            ):
    '''
    Pre : gives a numpy array of float represents x_train, a numpy array
    of float represents label for the corresponding x_train, a numpy array
    of string represents label names, a numpy array of float represents
    a single testing instance, an integer represents number of
    nearest neighbors needed (k) with default = 5, a string represents
    distance function with default = 'norm_k' (other: 'cosine_distance'),
    an integer represents the value of norm with default = 2.

    Post: returns a list of tuples represents k nearest neighbors for the
    testing instance in all training instance. The first element in each
    tuple is the label as string and the second element is the distance
    as float.
    '''
    test = testing_instance.reshape(-1,)
    distances = []

    for i in range(len(x_train)):
        instance = x_train[i].reshape(-1,)
        if distance_function == 'cosine_distance':
            distance = cosine_distance(test, instance)
        else:
            distance = norm_k(test, instance, norm)

        distances.append((label_names[y_train[i]], distance))

    return sorted(distances, key=itemgetter(1))[0:k]


def get_result(nn_list, verbose=0):
    '''
    Pre : gives a list of strings represents the nearest neighbors and an
    integer (0 or 1) represents the verbose.

    Post: if verbose != 1: returns the nearest neighbor as string; otherwise
    returns the list of all labels (string) appeared in k-nearest neighbors
    that sorted by the scores of each label.
    '''
    instances_dict = {}
    for instance in nn_list:
        if instance[0] not in instances_dict:
            instances_dict[instance[0]] = []
        instances_dict[instance[0]].append(instance[1])

    instances_result_list = []
    for instance in instances_dict:
        score = len(instances_dict[instance]) / len(nn_list)
        instances_result_list.append((instance, score))

    instances_result_list = sorted(
        instances_result_list, key=itemgetter(1), reverse=True
    )
    if verbose == 1:
        return instances_result_list
    return instances_result_list[0][0]


def predict(x_train, y_train, x_test, y_test, label_names):
    '''
    Pre : gives a numpy array of float represents x_train, a numpy array of
    float represents label indexes for the corresponding x_train, a numpy array
    of float represents all testing instances, a numpy array of float
    represents all labels indexes for testing instances, and a numpy array of
    string represents label's names for given data.

    Post: returns a DataFrame as the result of predictions with 3 columns:
    'Instance ID', 'Predict', 'Real Label'.
    '''
    right = 0
    total = 0

    result = pd.DataFrame(
        index=np.arange(0, len(x_test)),
        columns=['Instance ID', 'Predict', 'Real Label']
    )

    for i in range(len(x_test)):
        nearest_neighbors = get_k_nearest_neighbors(
            x_train, y_train, label_names, x_test[i], 10)
        y_pred = get_result(nearest_neighbors)
        if (y_pred == label_names[y_test[i]]):
            right += 1
        total += 1
        result.loc[i] = [i, y_pred, label_names[y_test[i]]]

        print('Instance ID: ', i, ';   Predict: ', y_pred,
              ';   Real Label: ', label_names[y_test[i]])
    print()
    print('Correct Prediction: ' + str(right))
    print('Total Prediction:' + str(total))
    print('Accuracy: ' + str(right / total * 100))
    print()
    return result


def visualize_images(data, labels, label_names, predict=None, channels=3,
                     start=0, cols=4, rows=4, size=10, fontsize=10):
    '''
    Pre : gives a numpy array represents data of color images (4 dimensions
    array), a numpy array represents labels for the corresponding images, a
    numpy array represents the label names, a numpy array represents the
    prediction for the given images, an integer represents number of channels
    of the given images with default = 3, an integer represents start index
    for visualization with default = 0, an integer represents number of
    columns with default = 4, an integer represents number of columns with
    default = 4, an integer represents size of image with default = 10, an
    integer represents size of title's font with default = 10.

    Post: plots predicted images and save the plot to
    'kNN predictions Examples.png'.
    '''
    if (channels != 3):
        data = data[:, :, :, 0]
    fig = plt.figure(figsize=(size, size))
    plt.subplots_adjust(bottom=.05, top=.95, hspace=.9)

    cols = cols
    rows = rows
    for i in range(1, cols * rows + 1):
        img = data[start + i - 1]
        fig.add_subplot(rows, cols, i)
        if (channels != 3):
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        if predict is not None:
            pred = predict[start + i - 1]
        else:
            pred = 'NaN'
        real = label_names[labels[start + i - 1]]
        plt.title('Predict: ' + pred + '\n Real: ' + real, fontsize=fontsize)
        plt.axis('off')
    plt.savefig('kNN Predictions Examples.png')
    plt.show()


def check_predictions_probability(
    x_train, y_train, x_test, y_test, label_names, number_instances=10
):
    '''
    Pre : gives a numpy array of float represents x_train, a numpy array
    of float represents label indexes for the corresponding x_train, a numpy
    array of float represents all testing instances, a numpy array of float
    represents labels indexes for testing instances, a numpy array of string
    represents label's names for given data, and an integer represents the
    number of instances to check with default = 10.

    Post: plots several predictions with probabilities of each label and
    saves the plot as png files.
    '''
    print('Use several examples to see how the model classify:')

    for i in range(number_instances):
        nearest_neighbors = get_k_nearest_neighbors(
            x_train, y_train, label_names, x_test[i], 10)
        scores = get_result(nearest_neighbors, verbose=1)

        scores_list = [0.0] * 10
        for item in scores:
            label = item[0]
            scores_list[int(np.where(label_names == label)[0])] = item[1]
        data = pd.DataFrame({'Labels': label_names, 'Scores': scores_list})
        sns.catplot(
                x='Labels',
                y='Scores',
                kind='bar',
                data=data
            )
        plt.xticks(rotation=-45)
        plt.title('Testing Instance ID: ' + str(i) + '\n'
                  'Predicted:' + str(scores[0][0])
                  + '\nReal Label:' + str(label_names[y_test[i]]))
        plt.savefig('kNN prediction probability instance ' + str(i) + '.png',
                    bbox_inches='tight')


def main():
    x_train = np.load('Models/x_train.npy')
    x_val = np.load('Models/x_val.npy')
    x_test = np.load('Models/x_test.npy')
    y_train = np.load('Models/y_train.npy')
    y_val = np.load('Models/y_val.npy')
    y_test = np.load('Models/y_test.npy')
    label_names = np.load('Models/label_names.npy')

    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val))

    # training part (remove comments notations to re-predicting)
    '''
    print('Testing model')
    testing_result = predict(x_train, y_train, x_test, y_test, label_names)
    testing_result.to_csv('kNN results.csv')
    '''

    # check predictions probability
    sns.set()
    check_predictions_probability(x_train, y_train, x_test, y_test,
                                  label_names)

    # correct predictions statistics
    result = pd.read_csv('kNN results.csv')
    correct_predictions = len(
        result[result['Predict'] == result['Real Label']])
    total_predictions = len(result)
    print('Prediction Results: ')
    print('   Correct Predictions: ' + str(correct_predictions))
    print('   Total Predictions: ' + str(total_predictions))
    print('   Accuracy: '
          + str(correct_predictions / total_predictions * 100) + '%')
    print()

    # visualize several predictions
    y_pred = result['Predict'].values
    visualize_images(x_test, y_test, label_names, y_pred, channels=3,
                     start=0, cols=8, rows=8, fontsize=8)


if __name__ == "__main__":
    main()
