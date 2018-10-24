from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
from numpy import array
from math import e


def normalize(pattern, lower=K.variable(0.0), upper=K.variable(1.0), epsilon=K.epsilon()):
    # assert lower < upper

    # Get the range of existing values
    R = K.max(pattern) - K.min(pattern)

    # Multiple the pattern by the scaling factor
    C = (upper - lower) / R
    scaled = C * pattern

    # Return the scaled values offset by the lower bound - minimum of the scaled values.
    result = scaled + (lower - K.min(scaled))
    return result


def gompertz(age, shape=5):
    """
    Our gompertz function assumes time
    values between 0 and 1.
    """
    t = (age * 2) - 1
    # e ^ -e ^ -(shape * t)
    return K.exp(-K.exp(-(shape * t)))



def apoptosis(p, age, turnover):
    dims = list(range(len(K.int_shape(p)) - 1))
    diffs = normalize(K.std(p, axis=dims)) * 0.65
    act = normalize(K.mean(K.abs(p), axis=dims)) * 0.15

    # rankings should also be a value between 0.0 and 1.0
    rankings = diffs + act + (age * 0.2)

    # Now pseudo-randomly reset a small percentage of the
    # e.g., keep_prob + rankings + rand / 3
    rand_tensor = 1.0 - turnover
    rand_tensor += K.random_uniform_variable(shape=K.shape(age), low=0.0, high=1.0)
    rand_tensor += rankings
    rand_tensor /= 3.0
    binary_tensor = K.round(rand_tensor)
    return age * binary_tensor

class Neurogen(Optimizer):
    """Neurogenesis optimizer

    # Arguments
        lr: Array of min and max learning rates
        momentum: Not implemented yet
        decay: Array of min and max decay rate
        sparsity: Array of min and max sparsity constraint
        turnover: Rate of neuronal turnover
        growth: Growth rate of neurons
    """
    def __init__(
        self,
        lr=array([0.01, 0.1]),
        momentum=0.0,
        decay=array([0.0, 0.0005]),
        sparsity=array([0.0, 0.1]),
        turnover=0.1,
        growth=0.05,
        **kwargs
    ):
        super(Neurogen, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr_min = K.variable(lr[0], name='lr-min')
            self.lr_max = K.variable(lr[1], name='lr-max')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay_min = K.variable(decay[0], name='decay-min')
            self.decay_max = K.variable(decay[1], name='decay-max')
            self.sparsity_min = K.variable(sparsity[0], name='sparsity-min')
            self.sparsity_max = K.variable(sparsity[1], name='sparsity-max')
            self.turnover = K.variable(turnover, name='turnover')
            self.growth = K.variable(growth, name='growth')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(s) for s in shapes]
        self.weights = [self.iterations] + moments

        # I'm not sure if this is the correct way of extract
        # this hidden layer dimension
        hsizes = [s[-1] for s in shapes]
        age = [K.random_uniform_variable(shape=(n,), low=0.0, high=1.0) for n in hsizes]

        for p, g, m, a in zip(params, grads, moments, age):
            # First compute the growth and lr, decay
            growth = gompertz(a)

            # Compute lr, decay, sparsity from age
            lr = self.compute_lr(growth)
            decay = self.compute_decay(growth)
            sparsity = self.compute_sparsity(growth)

            # Normal SGD
            # Compute the update
            v = self.momentum * m - lr * g - self.decay_max * p
            self.updates.append(K.update(m, v))
            new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            # Increase ages
            new_a = K.clip(a + self.growth, 0.0, 1.0)

            # Apoptosis (replacement)
            new_a = apoptosis(p, a, self.turnover)

            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(a, new_a))

        ######## Basic SGD for debugging ########
        # lr = self.lr_max

        # for p, g, m in zip(params, grads, moments):
        #     v = self.momentum * m - lr * g  # velocity
        #     self.updates.append(K.update(m, v))
        #     new_p = p + v

        #     # Apply constraints.
        #     if getattr(p, 'constraint', None) is not None:
        #         new_p = p.constraint(new_p)

        #     self.updates.append(K.update(p, new_p))

        return self.updates

    def compute_lr(self, growth):
        return normalize(1 - growth, lower=self.lr_min, upper=self.lr_max)

    def compute_decay(self, growth):
        return normalize(1 - growth, lower=self.decay_min, upper=self.decay_max)

    def compute_sparsity(self, growth):
        return normalize(growth, lower=self.sparsity_min, upper=self.sparsity_max)


if __name__ == '__main__':
    '''Trains a simple convnet on the MNIST dataset.
    Gets to 99.25% test accuracy after 12 epochs
    (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
    '''

    batch_size = 100
    num_classes = 10
    epochs = 3

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape,
            use_bias=False,
        )
    )
    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=False))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', use_bias=False))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', use_bias=False))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        # optimizer=keras.optimizers.Adadelta(),
        # optimizer=keras.optimizers.SGD(),
        optimizer=Neurogen(),
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
