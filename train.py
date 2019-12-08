"""
auther: leechh
"""
import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tqdm import tqdm
import matplotlib.pyplot as plt

from fiat.component.chunk import chunk
from fiat.logging import logging
from fiat.component.path import mkdir
from fiat.losses import lossfromname
from fiat.metrics import metricsfromname
from fiat.optimizer import optfromname
from fiat.data import TFR


class MeanCal(object):
    def __init__(self):
        self.count = 0
        self.sum = 0

    def update(self, value, batch):
        self.sum += (float(value) * batch)
        self.count += batch

    def cal(self):
        mean = np.NaN if self.count == 0 else self.sum / self.count
        return mean


class Model(object):
    """
    fiat 的一大特点既将模型实时的本地化, 这样的好处是在长时间的训练中如若遇到突发情况, 比如断电, 可以随时恢复, 最多损失一个 epoch，
    并且将一个模型的本地化路径作为该模型的“key”，该本地化路径会保存以下信息：
    model_path
        |-- ckpt
            |-- checkpoint
            |-- model.ckpt.data-00000-of-00001
            |-- model.ckpt.index
        |-- model.minfo
    ckpt文件夹，是一个标准的tensorflow checkpoint 保存文件夹。
    model.minfo 文件，保存着三个信息，该模型训练过程中的training loss, validition metrics, 以及 best epoch.
    """
    def __init__(self, path):
        self.path = path
        self.ckpt = os.path.join(path, 'ckpt')
        self.minfo = os.path.join(path, 'model.minfo')

    def __write_minfo(self, recorder):
        with open(self.minfo, 'w') as file:
            for key, val in recorder.items():
                file.write(f'{key}={val}\n')
                
    def copy(self, path):
        shutil.copytree(self.path, path)

    def write(self, recorder, sess):
        # new path
        mkdir(self.path)
        # write minfo
        self.__write_minfo(recorder)

        # write ckpt
        self.saver(False).save(sess=sess,
                               save_path=os.path.join(self.ckpt, 'model.ckpt'),
                               write_meta_graph=False)

    def clear_minfo(self):
        recorder = self.empty_recorder()
        self.__write_minfo(recorder)

    def rm(self):
        shutil.rmtree(self.path)

    def misdel_protector(self, retrain):
        if self.isexist() and not retrain:
            print(f'path {self.path} is exist, do you wish drop it? [Yes/No]')
            if input() == 'No':
                raise FileExistsError(f'path {self.path} is exist')

    def isexist(self):
        return os.path.exists(self.path)

    def __load_minfo(self):
        recorder = {}
        with open(self.minfo) as file:
            for line in file:
                line = line.strip().split('=')
                recorder[line[0]] = eval(line[1])
        return recorder

    def load(self, retrain, sess):
        if retrain:
            if self.isexist():
                # read minfo
                recorder = self.__load_minfo()

                # read ckpt
                self.saver(True).restore(sess=sess, save_path=os.path.join(self.ckpt, 'model.ckpt'))
                return recorder
            else:
                raise FileNotFoundError(f'path {self.path} is not exist')
        else:
            return self.empty_recorder()
        
    def readckpt(self):
        checkpoint_path = os.path.join(self.ckpt, 'model.ckpt')
        if self.isexist():
            reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return set(var_to_shape_map.keys())
        else:
            raise FileNotFoundError(f'ckpt file {checkpoint_path} is not exist')

    @staticmethod
    def empty_recorder():
        return {'train': [2 ** 32], 'valid': [-2 ** 32], 'now_epoch': 0}

    @staticmethod
    def cal_loss(loss, y_true, y_pred):
        with tf.name_scope('loss'):
            if type(loss) is str:
                return lossfromname(loss, y_true, y_pred)
            else:
                return loss(y_true, y_pred)

    @staticmethod
    def cal_metrics(metrics, y_true, y_pred):
        with tf.name_scope('metrics'):
            if type(metrics) is str:
                return metricsfromname(metrics, y_true, y_pred)
            else:
                return metrics(y_true, y_pred)

    @staticmethod
    def optimizer(optimizer, rate, loss):
        with tf.name_scope('optimizer'):
            if type(optimizer) is str:
                return optfromname(optimizer, learning_rate=rate, loss=loss)
            else:
                return optimizer(rate, loss)

    @staticmethod
    def tfr(data, batch_size, epoch, reshape, method):
        with tf.name_scope('data'):

            # build dataset & batch
            if type(data) is dict:
                for key, val in data.items():
                    data[key] = val if batch_size == 0 else val.batch(batch_size)
                dataset = data['train'].concatenate(data['valid'])
            else:
                dataset = data if batch_size == 0 else data.batch(batch_size)
            dataset = dataset.repeat(epoch)

            iterator = dataset.make_one_shot_iterator()
            img, label = iterator.get_next()

            img = tf.image.resize(img, size=reshape[:2], method=method)
            img = tf.reshape(img, [-1, reshape[0], reshape[1], reshape[2]])
        return img, label

    def saver(self, isload):
        if isload:
            distrainable = [i for i in tf.global_variables() if i.name.split(':')[0] in self.readckpt()]
            return tf.train.Saver(distrainable, max_to_keep=2147483647)
        else:
            return tf.train.Saver(max_to_keep=2147483647)

    def fit(self, arch, data, loss, metric, optimizer,
            rate, epoch=1, batch_size=32, early_stopping=False,
            verbose=2, retrain=False, reshape=None,
            reshape_method=None, distrainable=False):
        """

        :param arch:
        :param data:
        :param loss:
        :param metric:
        :param optimizer:
        :param rate:
        :param epoch:
        :param batch_size:
        :param early_stopping:
        :param verbose:
        :param retrain:
        :param reshape:
        :param reshape_method:
        :param distrainable:
        :return:
        """

        # reset graph
        tf.reset_default_graph()
        self.misdel_protector(retrain)

        # data
        data_dict, traincount, validcount = data()
        img, label = self.tfr(data_dict, batch_size, epoch, reshape, reshape_method)

        # arch
        y_true, y_pred = label, arch(img)
        y_true_shape, y_pred_shape = y_true.get_shape(), y_pred.get_shape()
        if y_true_shape != y_pred_shape:
            try:
                y_true = tf.image.resize(y_true, size=y_pred_shape[1: 3], method=reshape_method)
            except ValueError:
                pass

        # opt
        loss_tensor = self.cal_loss(loss, y_true, y_pred)
        metric_tensor = self.cal_metrics(metric, y_true, y_pred)
        opt = self.optimizer(optimizer, rate, loss_tensor)

        # sess
        error = 0
        with tf.Session() as sess:
            # init
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
            recorder = self.load(retrain, sess)

            # train
            start = recorder['now_epoch'] + 1
            for epoch_num in range(start, epoch + start):

                # split train valid
                cal_train, cal_valid = MeanCal(), MeanCal()

                train_chunk = chunk(traincount, batch_size)
                valid_chunk = chunk(validcount, batch_size)

                # tqdm
                pbar = tqdm(train_chunk) if verbose == 2 else train_chunk
                for batch in pbar:
                    # run training optimizer & get train loss
                    try:
                        _, loss_value = sess.run((opt, loss_tensor))
                        cal_train.update(loss_value, batch)
                    except tf.errors.OutOfRangeError:
                        break

                    # valid loss
                    last_chunk = (train_chunk.__len__() == 0)
                    if last_chunk:
                        for valid_batch in valid_chunk:
                            try:
                                metric_value = sess.run(metric_tensor)
                                cal_valid.update(metric_value, valid_batch)
                            except tf.errors.OutOfRangeError:
                                break

                    train_v, valid_v = cal_train.cal(), cal_valid.cal()
                    # description
                    if verbose == 2:
                        desc_train = train_v if len(train_chunk) == 0 else float(loss_value)
                        desc_str = f'epoch {epoch_num}, ' \
                                   f'train loss: {round(desc_train, 4)} ' \
                                   f'valid metric: {round(valid_v, 4)}'
                        pbar.set_description(desc_str)

                recorder['train'].append(train_v)
                recorder['valid'].append(valid_v)
                recorder['now_epoch'] = epoch_num

                # check point & early stopping
                if early_stopping:
                    if epoch_num == np.argmax(recorder['valid']):
                        error = 0
                        self.write(recorder, sess)
                    else:
                        error += 1
                        if error >= early_stopping:
                            break
                else:
                    self.write(recorder, sess)

        # final print
        recorder = self.__load_minfo()
        if verbose in [1, 2]:
            print(f'Final epoch is {recorder["now_epoch"]}, '
                  f' train score is {recorder["train"][-1]}, '
                  f'valid score is {recorder["valid"][-1]}')

        # write logging
        write_dict = {
            'id': (214013 * int(time.time()) + 2531011) % 2 ** 32,
            'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            'rate': rate,
            'recorder': recorder,
            'batch_size': batch_size,
            'reshape': reshape,
            'reshape_method': reshape_method,
            'model_path': self.path
        }
        log = logging()
        log.write(write_dict)
        return recorder

    def plot(self, start, count, arch, data, reshape, method=2, drop=False, rules=None):
        class2rgb = {0: [0], 1: [0, 1], 2: [1, 2], 3: [0, 2]}
        # clear graph
        tf.reset_default_graph()

        # TF dataset
        data_dict, traincount, validcount = data()
        img, label = self.tfr(data_dict, 1, 1, reshape, method)

        # arch
        y_true, y_pred = label, arch(img)
        y_true_shape, y_pred_shape = y_true.get_shape(), y_pred.get_shape()
        if y_true_shape != y_pred_shape:
            try:
                y_true = tf.image.resize(y_true, size=y_pred_shape[1: 3], method=method)
            except ValueError:
                pass

        # saver & sess
        saver = self.saver()
        with tf.Session() as sess:
            # init
            sess.run(tf.global_variables_initializer())

            if self.path is not None:
                self.load(False, sess)

            for idx in range(start + count):
                img_arr, y_true_arr, y_pred_arr = sess.run((img, y_true, y_pred))
                if idx >= start:
                    img_arr = np.clip(img_arr[0], 0, 255)
                    y_true_arr = np.clip(y_true_arr[0], 0, 1)
                    y_pred_arr = np.clip(y_pred_arr[0], 0, 1)
                    y_pred_arr = rules(y_pred_arr) if rules is not None else y_pred_arr
                    h, w, c = img_arr.shape
                    _, _, f = y_true_arr.shape

                    plt.figure(figsize=[20, 10])
                    # print y true
                    for i, arr in enumerate([y_true_arr, y_pred_arr]):
                        plt.subplot(1, 2, i + 1)
                        if (not drop) or (i is 0):
                            plt.imshow(img_arr.astype('int'))
                        for layer in range(f):
                            mask = np.ones([h, w, 3]) * 255
                            label = arr[:, :, layer][..., np.newaxis].astype('float')
                            mask[:, :, class2rgb[layer]] -= 160.
                            mask = np.concatenate([mask, label * 170], axis=2)
                            plt.imshow(mask.astype('int'))