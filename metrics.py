"""
auther: leechh
license: MIT
metrics - larger is better
"""
import tensorflow as tf


def tp(mean=False):

    def metrics(y_true, y_pred):
        """
        :param y_true: [0., 1.]
        :param y_pred: [0., 1.]
        """
        if not mean:
            return tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
        else:
            rank = tf.rank(y_true)
            return tf.math.reduce_mean(tf.math.reduce_sum(
                tf.math.multiply(y_true, y_pred), axis=list(range(int(rank - 1)))))
    return metrics


def fn(mean=False):

    def metrics(y_true, y_pred):
        """
        :param y_true: [0., 1.]
        :param y_pred: [0., 1.]
        """
        if not mean:
            return tf.math.reduce_sum(tf.dtypes.cast(tf.math.logical_and(
                tf.math.equal(y_true, 0.), tf.math.equal(y_pred, 1.)), dtype=tf.float32))
        else:
            rank = tf.rank(y_true)
            axis = list(range(int(rank - 1)))
            return tf.math.reduce_mean(tf.math.reduce_sum(tf.dtypes.cast(tf.math.logical_and(
                tf.math.equal(y_true, 0.), tf.math.equal(y_pred, 1.)), dtype=tf.float32), axis=axis))
    return metrics


def fp(mean=False):

    def metrics(y_true, y_pred):
        """
        :param y_true: range [0, 1]
        :param y_pred: range [0, 1]
        """
        if not mean:
            return tf.math.reduce_sum(tf.dtypes.cast(tf.math.logical_and(
                tf.math.equal(y_true, 1.), tf.math.equal(y_pred, 0.)), dtype=tf.float32))
        else:
            rank = tf.rank(y_true)
            axis = list(range(int(rank - 1)))
            return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.math.logical_and(
                tf.math.equal(y_true, 1.), tf.math.equal(y_pred, 0.)), dtype=tf.float32), axis=axis))
    return metrics


def tn(mean=False):

    def metrics(y_true, y_pred):
        """
        :param y_true: range [0, 1]
        :param y_pred: range [0, 1]
        """
        if not mean:
            return tf.math.reduce_sum(tf.dtypes.cast(tf.math.logical_and(
                tf.math.equal(y_true, 0.), tf.math.equal(y_pred, 0.)), dtype=tf.float32))
        else:
            rank = tf.rank(y_true)
            axis = list(range(int(rank - 1)))
            return tf.math.reduce_mean(tf.math.reduce_sum(tf.dtypes.cast(tf.math.logical_and(
                tf.math.equal(y_true, 0.), tf.math.equal(y_pred, 0.)), dtype=tf.float32), axis=axis))
    return metrics


def dice(threshold=0.5, smooth=1):

    def metrics(y_true, y_pred):
        # y to 0. & 1.
        y_pred = tf.dtypes.cast(tf.math.greater(y_pred, threshold), tf.float32)
        y_true = tf.dtypes.cast(tf.math.greater(y_true, threshold), tf.float32)
        # dice
        intersection = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0, 1, 2])
        union = tf.math.reduce_sum(y_true, axis=[0, 1, 2]) + tf.math.reduce_sum(y_pred, axis=[0, 1, 2])
        return tf.math.reduce_mean(((2 * intersection) + smooth) / (union + smooth))
    return metrics


def neg_mse():

    def metrics(y_true, y_pred):
        return - tf.math.reduce_mean((y_pred - y_true)**2)
    return metrics


def neg_mae():

    def metrics(y_true, y_pred):
        return - tf.math.reduce_mean(tf.math.abs(y_pred - y_true))
    return metrics


def accuracy(threshold=0.5):

    def metrics(y_true, y_pred):
        y_pred = tf.dtypes.cast(tf.math.greater(y_pred, threshold), tf.float32)
        y_true = tf.dtypes.cast(tf.math.greater(y_true, threshold), tf.float32)
        return tf.math.reduce_mean(tf.dtypes.cast(tf.math.equal(y_true, y_pred), tf.float32))
    return metrics


def precision(threshold=0.5):

    def metrics(y_true, y_pred):
        y_true = tf.dtypes.cast(tf.math.greater(y_true, threshold), tf.float32)
        y_pred = tf.dtypes.cast(tf.math.greater(y_pred, threshold), tf.float32)
        return tp()(y_true, y_pred) / (tp()(y_true, y_pred) + fp()(y_true, y_pred))
    return metrics


def micro_precision(threshold=0.5):

    def metrics(y_true, y_pred):
        y_true = tf.dtypes.cast(tf.math.greater(y_true, threshold), tf.float32)
        y_pred = tf.dtypes.cast(tf.math.greater(y_pred, threshold), tf.float32)
        return tp(True)(y_true, y_pred) / (tp(True)(y_true, y_pred) + fp(True)(y_true, y_pred))
    return metrics


def recall(threshold=0.5):

    def metrics(y_true, y_pred):
        y_true = tf.dtypes.cast(tf.greater(y_true, threshold), tf.float32)
        y_pred = tf.dtypes.cast(tf.greater(y_pred, threshold), tf.float32)
        return tp()(y_true, y_pred) / (tp()(y_true, y_pred) + fn()(y_true, y_pred))
    return metrics


def micro_recall(threshold=0.5):

    def metrics(y_true, y_pred):
        y_true = tf.dtypes.cast(tf.math.greater(y_true, threshold), tf.float32)
        y_pred = tf.dtypes.cast(tf.math.greater(y_pred, threshold), tf.float32)
        return tp(True)(y_true, y_pred) / (tp(True)(y_true, y_pred) + fn(True)(y_true, y_pred))
    return metrics


def f1(threshold=0.5, beta=1, smooth=1e-6):
    def metrics(y_true, y_pred):
        return (1 + beta**2) * precision(threshold)(y_true, y_pred) * recall(threshold)(y_true, y_pred) /\
               (((beta**2) * precision(threshold)(y_true, y_pred)) + recall(threshold)(y_true, y_pred))
    return metrics


def micro_f1(threshold=0.5, beta=1):
    def metrics(y_true, y_pred):
        return (1 + beta**2) * micro_precision(threshold)(y_true, y_pred) * micro_recall(threshold)(y_true, y_pred) /\
               (((beta**2) * micro_precision(threshold)(y_true, y_pred)) + micro_recall(threshold)(y_true, y_pred))
    return metrics


def metricsfromname(metrics_name, y_true, y_pred):
    # y_*: [batch, height, width, num_class]
    # dice
    if metrics_name == 'dice':
        return dice()(y_true, y_pred)
    
    elif metrics_name == 'accuracy':
        return accuracy()(y_true, y_pred)

    elif metrics_name == 'precision':
        return precision()(y_true, y_pred)

    elif metrics_name == 'micro_precision':
        return micro_precision()(y_true, y_pred)

    elif metrics_name == 'recall':
        return recall()(y_true, y_pred)

    elif metrics_name == 'micro_recall':
        return micro_recall()(y_true, y_pred)

    elif metrics_name == 'f1':
        return f1()(y_true, y_pred)

    elif metrics_name == 'micro_f1':
        return micro_f1()(y_true, y_pred)
    
    elif metrics_name == 'neg_mse':
        return neg_mse()(y_true, y_pred)

    elif metrics_name == 'neg_mae':
        return neg_mae()(y_true, y_pred)

    else:
        raise ValueError(f'metrics function name {metrics_name} ISNOT exist')