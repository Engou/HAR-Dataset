#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/4/19 17:13
    @author: liyakun
    The file is a old way for process wisdm data
"""
from __future__ import division
import numpy as np
import os
import random
import tqdm


def check_balance_data(train_x, train_y):
    assert train_x.shape[0] == train_y.shape[0]
    sample_num = len(train_y)
    class_total = []
    for i in range(sample_num):
        class_total.append(train_y[i, :].tolist()[0])
    class_name = list(set(class_total))
    class_num_dic = {}
    for i in class_name:
        class_num_dic[i] = class_total.count(i) / sample_num
    for name, rate in class_num_dic.items():
        print('%s rate is:%s' % (str(name), str(rate)))
    print('the sample total num is:%d' % sample_num)


def shuffle_data(train_x, train_y):
    data_num = len(train_x)
    list_type = 1
    if isinstance(train_x, list):
        train_x = np.asarray(train_x)
        list_type = 1
    else:
        list_type = 0

    if isinstance(train_y, list):
        train_y = np.asarray(train_y)
        list_type = 1
    else:
        list_type = 0
    index = [i for i in range(data_num)]
    random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]
    if list_type == 1:
        return train_x.tolist(), train_y.tolist()
    return train_x, train_y


def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def normalization(data):
    '''
    data normalization
    :param data:
    :return:
    '''
    data = data.astype('float32')
    data = (data - np.min(data, axis=1, keepdims=True)) / (
        np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True) + 1e-5)
    return data


def load_X(file_name):
    X_signals = []
    for signal_type_path in file_name:
        f = open(signal_type_path, 'r')
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [row.replace('  ', ' ').strip().split(' ') for row in f]])
        f.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], dtype=np.int32)
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


def load_UCI_data(data_path=r'/data/UCI HAR Dataset/',
                  export_dir=r'/data/UCI HAR Dataset',
                  train_x_name=r'train_x.npy',
                  train_y_name=r'train_y.npy',
                  test_x_name=r'test_x.npy',
                  test_y_name=r'test_y.npy',
                  train_data_shuffle=True,
                  use_one_hot=True,
                  use_normalization=True,
                  expanding_data=False,
                  check_balance=True,
                  split_rate=0,
                  plt_data=True):
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"]

    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"]

    TRAIN_DTAT_PATH = data_path + r'/train/Inertial Signals'
    train_y_path = data_path + r'/train/y_train.txt'

    TEST_DATA_PATH = data_path + r'/test/Inertial Signals'
    test_y_path = data_path + r'/test/y_test.txt'

    train_x_path = [os.path.join(TRAIN_DTAT_PATH, file_name + 'train.txt') for file_name in INPUT_SIGNAL_TYPES]
    test_x_path = [os.path.join(TEST_DATA_PATH, file_name + 'test.txt') for file_name in INPUT_SIGNAL_TYPES]

    if not os.path.exists(train_x_name) or not os.path.exists(train_y_name) or not os.path.exists(
            test_x_name) or not os.path.exists(test_y_name):
        x_train = load_X(train_x_path)
        x_test = load_X(test_x_path)
        y_train = load_y(train_y_path)
        y_test = load_y(test_y_path)
        total_x = np.concatenate((x_train, x_test))
        total_y = np.concatenate((y_train, y_test))
        if split_rate == 0:
            len_train = len(x_train)
            len_test = len(x_test)
            total_x, total_y = shuffle_data(total_x, total_y)
            x_train = total_x[len_test:]
            y_train = total_y[len_test:]
            x_test = total_x[:len_test]
            y_test = total_y[:len_test]
        else:
            split_point = int(len(total_x) * split_rate)
            x_test = total_x[:split_point]
            y_test = total_y[:split_point]
            x_train = total_x[split_point:]
            y_train = total_y[split_point:]

        if export_dir is None:
            export_dir = os.path.abspath(os.path.dirname(train_x_name) + os.path.sep + ".")
        np.save(train_x_name, x_train)
        np.save(test_x_name, x_test)
        np.save(train_y_name, y_train)
        np.save(test_y_name, y_test)

    else:
        x_train, y_train = np.load(train_x_name), np.load(train_y_name)
        x_test, y_test = np.load(test_x_name), np.load(test_y_name)
    if check_balance:
        check_balance_data(x_train, y_train)

    if use_one_hot:
        print('Start One Hot Encode.....')
        y_train = one_hot(y_train)
        y_test = one_hot(y_test)

    if use_normalization:
        print('Start normalization.....')
        x_train = normalization(x_train)
        x_test = normalization(x_test)

    if train_data_shuffle:
        print('Start shuffle for trian data.....')
        data_num = len(x_train)
        index = [i for i in range(data_num)]
        random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]

    if expanding_data:
        print('Start expanding for data.....')
        x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
        x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])
        # x_train = x_train.reshape((-1, 1, x_train.shape[1], x_train.shape[2]))
        # x_test = x_train.reshape((-1, 1, x_test.shape[1], x_test.shape[2]))
        print('expanded train_x shape is:', x_train.shape)
        print('expanded train_y shape is:', y_train.shape)
        print('expanded test_x shape is:', x_test.shape)
        print('expanded test_y shape is:', y_test.shape)
    else:
        print('no expand train_x shape is:', x_train.shape)
        print('no expand train_y shape is:', y_train.shape)
        print('no expand test_x shape is:', x_test.shape)
        print('no expand test_y shape is:', y_test.shape)
    return x_train, y_train, x_test, y_test, LABELS


if __name__ == '__main__':
    data_path = r'G:/data/UCI HAR Dataset/'
    export_dir = 'G:/data/prepared_data/UCI'
    train_x_name = r'G:/data/Prepared_Data/UCI/train_x.npy'
    train_y_name = r'G:/data/Prepared_Data/UCI/train_y.npy'
    test_x_name = r'G:/data/Prepared_Data/UCI/test_x.npy'
    test_y_name = r'G:/data/Prepared_Data/UCI/test_y.npy'

    X_train, y_train, X_test, y_test, labels = load_UCI_data(data_path=data_path,
                                                             export_dir=export_dir,
                                                             train_x_name=train_x_name,
                                                             train_y_name=train_y_name,
                                                             test_x_name=test_x_name,
                                                             test_y_name=test_y_name,
                                                             train_data_shuffle=True,
                                                             use_one_hot=True,
                                                             use_normalization=True,
                                                             expanding_data=True,
                                                             check_balance=True)

    print('Train datas shape:', X_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test datas shape:', X_test.shape)
    print('Test labels shape:', y_test.shape)
