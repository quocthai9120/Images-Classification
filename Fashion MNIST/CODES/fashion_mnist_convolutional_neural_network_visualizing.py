# Student Name: Thai Quoc Hoang - uwnetid: qthai912
# Student ID: 1861609
# Section: CSE163 AC
# Instructor: Hunter Schafer

# Program Description: This program does the visulazation for CNN approach
# with Fashion MNIST dataset.


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.utils import to_categorical


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

    Post: plots predicted images and save the plot to 'CNN predictions.png'.
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
            pred = label_names[predict[start + i - 1]]
        else:
            pred = 'NaN'
        real = label_names[int(np.where(labels[start + i - 1] == 1)[0])]
        plt.title('Predict: ' + pred + '\n Real: ' + real, fontsize=fontsize)
        plt.axis('off')
    plt.savefig('CNN predictions.png')
    plt.show()


def visualize_filters(model, layer_index, channel=0, figsize=(5, 5)):
    '''
    Pre : gives a Keras sequential model, an integer represents the index
    of layer, an integer represent index of channel with default = 0, a
    tuple represents figure size with default = (5, 5).

    Post: plots predicted filters at the given channel and save the plots
    as png files.
    '''
    filters = model.layers[layer_index].get_weights()[0][:, :, :, :][:, :, :]
    rows = 4 * (filters.shape[3] // 32)
    cols = filters.shape[3] // rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(bottom=.05, top=.95, hspace=.9)
    index = 0

    for i in range(rows):
        for j in range(cols):
            # filters[row, col, channel,filters index]
            axes[i, j].imshow(filters[:, :, channel, index], cmap='gray')
            index += 1
            axes[i, j].axis('off')
    title = 'Filters visualization - Layer: ' + str(layer_index)\
            + ' - channel: ' + str(channel)
    fig.suptitle(title)
    plt.savefig(title + '.png')
    plt.show()


def visualize_activations(model, data, image_index, layer):
    '''
    Pre : gives a Keras Sequential model, a numpy array (4D) represents images,
    an integer represents the image's index needed to visualize, an integer
    represents the layer that need to visualize.

    Post: plots predicted activations at the given channel and save the plots
    as png files.
    '''
    activations = Model(inputs=model.input, outputs=model.layers[layer].output)

    number_filters = int(model.layers[layer].output.shape[3])
    rows = 4 * (number_filters // 32)
    cols = number_filters // rows
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    index = 0
    for i in range(rows):
        for j in range(cols):
            image = activations.predict(
                data[image_index:image_index+1])[0, :, :, index]
            axes[i][j].imshow(image, cmap='gray')
            axes[i][j].axis('off')
            index += 1
    title = 'Activation visualization Layer ' + str(layer)
    fig.suptitle(title)
    plt.savefig(title + '.png')
    plt.show()


def main():
    # load model and data
    model = keras.models.load_model('Models/cnn_model')
    x_test = np.load('Models/x_test.npy').astype('float32') / 255
    y_test = np.load('Models/y_test.npy')
    label_names = np.load('Models/label_names.npy')
    num_classes = len(label_names)
    channels = 1

    # put y_test to categories
    y_test = to_categorical(y_test, num_classes)

    # visualize predictions
    y_pred = model.predict_classes(x_test)
    visualize_images(x_test, y_test, label_names, y_pred, channels=channels,
                     start=0, cols=8, rows=8, fontsize=8)

    # print model summary
    print()
    print('Model summary:')
    print(model.summary())
    print()

    # visualize filters of the first 3 channel for each convolutional layer
    for layer_index in [0, 3, 5]:
        for i in range(channels):
            visualize_filters(model, layer_index, channel=i, figsize=(5, 5))

    # visualize outputs after each convolutional layer
    for i in range(8):
        visualize_activations(model, x_test, image_index=16, layer=i)


if __name__ == "__main__":
    main()
