# Student Name: Thai Quoc Hoang - uwnetid: qthai912
# Student ID: 1861609
# Section: CSE163 AC
# Instructor: Hunter Schafer

# Program Description: This program calls all other files to initialize data
# then runs both models kNN and CNN for the CIFAR_10 dataset.


import cifar_10_data_preprocessing
import cifar_10_k_nearest_neighbors
import cifar_10_convolutional_neural_network_training
import cifar_10_convolutional_neural_network_visualizing


def main():
    cifar_10_data_preprocessing.main()
    cifar_10_k_nearest_neighbors.main()
    cifar_10_convolutional_neural_network_training.main()
    cifar_10_convolutional_neural_network_visualizing.main()


if __name__ == "__main__":
    main()
