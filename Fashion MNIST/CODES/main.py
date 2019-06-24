# Student Name: Thai Quoc Hoang - uwnetid: qthai912
# Student ID: 1861609
# Section: CSE163 AC
# Instructor: Hunter Schafer

# Program Description: This program calls all other files to initialize data
# then runs both models kNN and CNN for the Fashion MNIST dataset.

import fashion_mnist_data_preprocessing as init
import fashion_mnist_k_nearest_neighbors as kNN
import fashion_mnist_convolutional_neural_network_training as cnn_training
import fashion_mnist_convolutional_neural_network_visualizing as cnn_visualize


def main():
    init.main()
    kNN.main()
    cnn_training.main()
    cnn_visualize.main()


if __name__ == "__main__":
    main()
