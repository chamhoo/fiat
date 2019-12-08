"""
auther: leechh
license: MIT
losses - lower is better
"""
import tensorflow as tf


def bce(sample_weight=None):
    def loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred, sample_weight)
    return loss


def mse():
    def loss(y_true, y_pred):
        return tf.math.reduce_mean((y_true - y_pred) ** 2)
    return loss


def mae():
    def loss(y_true, y_pred):
        return tf.math.reduce_mean(tf.abs(y_true - y_pred))
    return loss


def dice(axis=None):
    axis = [1, 2] if axis is None else axis

    def loss(y_true, y_pred):
        smooth = 1
        y_true = tf.clip_by_value(y_true, 0, 1)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        y_true = tf.keras.layers.Flatten()(y_true)
        intersection = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=None)
        union = tf.math.reduce_sum(y_true, axis=None) + tf.math.reduce_sum(y_pred, axis=None)
        return 1. - tf.math.reduce_mean(((2 * intersection) + smooth) / (union + smooth))
    return loss


def jaccard(axis=None):
    axis = [0, 1, 2] if axis is None else axis

    def loss(y_true, y_pred):
        smooth = 1
        y_true = tf.clip_by_value(y_true, 0, 1)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        y_true = tf.keras.layers.Flatten()(y_true)
        intersection = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=None)
        union = tf.math.reduce_sum(y_true, axis=None) + tf.math.reduce_sum(y_pred, axis=None)
        return 1. - tf.math.reduce_mean((intersection + smooth) / ((union + smooth) - intersection))
    return loss


def lossfromname(loss_name, y_true, y_pred):
    # y_*: [batch, height, width, num_class]
    # dice
    if loss_name == 'dice':
        return dice()(y_true, y_pred)

    if loss_name == 'mse':
        return mse()(y_true, y_pred)

    if loss_name == 'mae':
        return mae()(y_true, y_pred)

    if loss_name == 'bce':
        return bce()(y_true, y_pred)

    if loss_name == 'jaccard':
        return jaccard()(y_true, y_pred)

    else:
        raise ValueError(f'loss name is in: dice, mse, mae, bce, jaccard. {loss_name} ISNOT exist')