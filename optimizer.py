"""
auther: leechh
"""
import tensorflow as tf


def adam(beta1=0.9, beta2=0.999, epsilon=1e-07):
    def opt(learning_rate, loss):
        return tf.keras.optimizers.adam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon).minimize(loss)
    return opt


def Adamax(beta1=0.9, beta2=0.999, epsilon=1e-07):
    def opt(learning_rate, loss):
        return tf.keras.optimizers.adam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon).minimize(loss)
    return opt


def Nadam(beta1=0.9, beta2=0.999, epsilon=1e-07):
    def opt(learning_rate, loss):
        return tf.keras.optimizers.adam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon).minimize(loss)
    return opt


def Adadelta(rho=0.95, epsilon=1e-07):
    def opt(learning_rate, loss):
        return tf.keras.optimizers.adam(
            learning_rate=learning_rate,
            rho=0.95,
            epsilon=epsilon).minimize(loss)
    return opt


def Adagrad(initial_accumulator_value=0.1, epsilon=1e-07):
    def opt(learning_rate, loss):
        return tf.keras.optimizers.adam(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon).minimize(loss)
    return opt


def sgd(momentum=0.99):
    def opt(learning_rate, loss):
        return tf.keras.optimizers.SGD(
            momentum=momentum,
            learning_rate=learning_rate).minimize(loss)
    return opt



def optfromname(name, learning_rate, loss):
    if name == 'adam':
        return adam()(learning_rate, loss)

    elif name == 'Adamax':
        return Adamax()(learning_rate, loss)

    elif name == 'Nadam':
        return Nadam()(learning_rate, loss)

    elif name == 'Adadelta':
        return Adadelta()(learning_rate, loss)

    elif name == 'Adagrad':
        return Adagrad()(learning_rate, loss)

    elif name == 'sgd':
        return sgd()(learning_rate, loss)
        
    else:
        assert False, 'optimizer name ISNOT exist'