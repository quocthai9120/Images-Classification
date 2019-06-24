# Student Name: Thai Quoc Hoang - uwnetid: qthai912
# Student ID: 1861609
# Section: CSE163 AC
# Instructor: Hunter Schafer

# Program Description: This program initializes data for the
# Fashion MNIST dataset then export to use the data later.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def main():
    # read data as DataFrame
    data = pd.read_csv('fashion-mnist_train.csv')
    test_data = pd.read_csv('fashion-mnist_test.csv')

    # get x_train, x_val, x_test, y_train, y_val, y_test as numpy arrays
    x = data.loc[:, data.columns != 'label']
    y = data.loc[:, data.columns == 'label']
    x_test = test_data.loc[:, data.columns != 'label']
    y_test = test_data.loc[:, data.columns == 'label']

    x = np.array(x).reshape(60000, 28, 28, 1)
    y = np.array(y).reshape(-1, )
    x_test = np.array(x_test).reshape(10000, 28, 28, 1)
    y_test = np.array(y_test).reshape(-1, )

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=91)

    # define labels as numpy array
    label_names = np.array([
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
        ])

    # export data
    os.mkdir('Models')
    save_dir = os.path.join(os.getcwd(), 'Models')
    np.save(save_dir + '/x_train', x_train)
    np.save(save_dir + '/x_val', x_val)
    np.save(save_dir + '/y_train', y_train)
    np.save(save_dir + '/y_val', y_val)
    np.save(save_dir + '/x_test', x_test)
    np.save(save_dir + '/y_test', y_test)
    np.save(save_dir + '/label_names', np.array(label_names))


if __name__ == "__main__":
    main()
