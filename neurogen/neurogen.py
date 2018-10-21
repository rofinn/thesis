from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Optimizer
from keras.legacy import interfaces
from numpy import array
from math import e

import keras.backend as K
import K.random_uniform_variable as uniform


def normalize(pattern, lower=K.variable(0.0), upper=K.variable(1.0), epsilon=K.epsilon()):
    assert lower < upper

    # Get the range of existing values
    R = max(pattern) - min(pattern)

    # Multiple the pattern by the scaling factor
    C = (upper - lower) / R
    scaled = C * pattern

    # Return the scaled values offset by the lower bound - minimum of the scaled values.
    result = scaled + (lower - min(scaled))
    return result


def gompertz(age, shape=5):
    """
    Our gompertz function assumes time
    values between 0 and 1.
    """
    t = (age * 2) - 1
    return e ^ -e ^ -(shape * t)


def apoptosis(p, age, turnover):
    diffs = normalize(K.std(p, axis=1)) * 0.65
    act = normalize(K.mean(K.abs(p), axis=1)) * 0.15

    # rankings should also be a value between 0.0 and 1.0
    rankings = nodiffs + act + (age * 0.2)

    # Now pseudo-randomly reset a small percentage of the
    # e.g., keep_prob + rankings + rand / 3
    rand_tensor = 1.0 - turnover
    rand_tensor += rankings
    rand_tensor += uniform(shape=K.shape(age), low=0.0, high=1.0) + rankings)
    rand_tensor /= 3.0
    binary_tensor = math_ops.floor(random_tensor)
    return age * binary_tensor

def Neurogen(Optimizer):
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
        lr=array([0.001, 0.1]),
        momentum=0.0,
        decay=array([0.0, 0.005]),
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
            self.decay_min = K.variable(lr[0], name='decay-min')
            self.decay_max = K.variable(lr[1], name='decay-max')
            self.sparsity_min = K.variable(lr[0], name='sparsity-min')
            self.sparsity_max = K.variable(lr[1], name='sparsity-max')
            self.turnover = K.variable(turnover, name='turnover')
            self.growth = K.variable(growth, name='growth')

    @interfaces.legacy_get_upates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = self.iterations + moments

        # I'm not sure if this is the correct way of extract
        # this hidden layer dimension
        print(shapes)
        hsizes = [s[1] for s in shapes]
        age = [uniform(shape=(n,), low=0.0, high=1.0) for n in hsizes]

        for p, g, a in zip(params, grads, moments, age):
            # First compute the growth and lr, decay
            growth = gompertz(a)

            # Compute lr, decay, sparsity from age
            lr = self.compute_lr(growth)
            decay = self.compute_decay(growth)
            sparsity = self.compute_sparsity(growth)

            # Normal SGD
            # Compute the update
            v = lr * g - decay * p
            new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            # Increase ages
            new_a = K.clip(a + self.growth, 0.0, 1.0)

            # Apoptosis (replacement)
            new_a = apoptosis(p, age, self.turnover)

            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(a, new_a))

        return self.updates

    def compute_lr(self, growth):
        return normalize(1 - growth, lower=self.lr_min, upper=self.lr_max)

    def compute_decay(self, growth):
        return normalize(1 - growth, lower=self.decay_min, upper=self.decay_max)

    def compute_sparsity(self, growth):
        return normalize(growth, lower=self.sparsity_min, upper=self.sparsity_max)


# model = Sequential()

# model.add(Dense(units=64, activation='relu', input_dim=100))
# model.add(Dense(units=10, activation='softmax'))

# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='sgd',
#     metrics=['accuracy']
# )

# model.fit(x_train, y_train, epochs=5, batch_size=32)
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# classes = model.predict(x_test, batch_size=128)
