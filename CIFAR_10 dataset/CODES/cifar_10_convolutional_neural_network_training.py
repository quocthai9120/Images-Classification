# Student Name: Thai Quoc Hoang - uwnetid: qthai912
# Student ID: 1861609
# Section: CSE163 AC
# Instructor: Hunter Schafer

# Program Description: This program does the training process for CNN approach
# with CIFAR_10 dataset.


from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def check_predictions_probability(model, x_test, y_test, label_names,
                                  number_instances=10):
    '''
    Pre : gives a Kereas Sequential model, a numpy array represents
    x_test with 4 dimensions (color images), a numpy array represents
    all labels for corresponding testing images, a numpy array of label
    names, and an integer represents number of testing instances with
    default = 10.

    Post: plots several predictions with probabilities of each label
    and save as png files.
    '''
    print('Use several examples to see how the model classify:')
    y_pred = model.predict(x_test)

    for i in range(number_instances):
        scores = y_pred[i]
        data = pd.DataFrame({'Labels': label_names, 'Scores': scores})
        sns.catplot(
                x='Labels',
                y='Scores',
                kind='bar',
                data=data
            )
        plt.xticks(rotation=-45)
        plt.title('Testing Instance ID: ' + str(i) + '\n'
                  'Predicted:' + str(label_names[np.argmax(scores)])
                  + '\nReal Label:'
                  + str(label_names[int(np.where(y_test[i] == 1)[0])]))
        plt.savefig('CNN prediction probability instance ' + str(i) + '.png',
                    bbox_inches='tight')


def save_model(model, name, direction):
    '''
    Pre : gives a Kereas Sequential model, a file name as string, and
    a direction from current directory as string

    Post: saves the model to the given path
    '''
    save_dir = os.path.join(os.getcwd(), direction)
    model_path = os.path.join(save_dir, name)
    model.save(model_path)


def main():
    # load data
    x_train = np.load('Models/x_train.npy').astype('float32') / 255
    x_val = np.load('Models/x_val.npy').astype('float32') / 255
    x_test = np.load('Models/x_test.npy').astype('float32') / 255
    y_train = np.load('Models/y_train.npy')
    y_val = np.load('Models/y_val.npy')
    y_test = np.load('Models/y_test.npy')
    label_names = np.load('Models/label_names.npy')
    num_classes = len(label_names)

    # put y_train, y_val, and y_test to categories
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # test the shape of data to verify correctness
    assert((45000, 32, 32, 3) == x_train.shape)
    assert((5000, 32, 32, 3) == x_val.shape)
    assert((10000, 32, 32, 3) == x_test.shape)
    assert((45000, 10) == y_train.shape)
    assert((5000, 10) == y_val.shape)
    assert((10000, 10) == y_test.shape)

    # initialize model
    model = Sequential()

    model.add(Conv2D(
        filters=32, kernel_size=(5, 5), input_shape=x_train.shape[1:]
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=64, kernel_size=(5, 5)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(
        optimizer='SGD',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # fit the model with training data and validate with validation data
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=25,
        validation_data=(x_val, y_val),
        shuffle=True
    )

    # plot the model's validation loss for further hyperparameters tuning
    val_loss = history.history['val_loss']
    plt.plot(np.arange(len(val_loss)), val_loss)
    plt.xlabel('epoch')
    plt.ylabel('validation loss')
    plt.title('CNN model validation loss history')
    plt.savefig('CNN model validation loss history.png')
    plt.show()

    # predict testing data
    print()
    print('Evaluating testing data:')
    scores = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )
    print('   Model accuracy for testing data:', str(scores[1] * 100) + '%')
    print()

    # check predictions probability
    sns.set()
    check_predictions_probability(model, x_test, y_test, label_names)

    # export model
    save_model(model, name='cnn_model', direction='Models')


if __name__ == "__main__":
    main()
