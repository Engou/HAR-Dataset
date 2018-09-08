#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/4/23 16:21
    @author: liyakun
    The file is a old way for process wisdm data
"""
from __future__ import division
import os
import numpy as np
import random
from skoda.pre_skoda_data import generate_data
from imblearn.combine import SMOTEENN
import platform
import pandas as pd
import json

LABELS = ['null class', 'write on notepad', 'open hood', 'close hood', 'check graps on the front door',
          'open left front door', 'close left front door', 'close booth left door', 'check trunk gaps',
          'open and close trunk', 'check steering wheel']


# np.random.seed(12345)


def balance_data(train_data, train_label):
    # small_calss_idx = [idx for idx in range(len(train_label)) if train_label[idx] != 0]
    # small_class_data = train_data[small_calss_idx]
    # small_class_label = train_label[small_calss_idx]
    # dif = len(train_data)-len(small_calss_idx)
    # copy_num = dif // len(small_calss_idx)
    # if dif > len(small_calss_idx)*3:
    #     for _ in range(copy_num):
    #         train_data = np.vstack([train_data, small_class_data])
    #         train_label = np.vstack([train_label, small_class_label])
    #     return train_data, train_label
    # else:
    #     return train_data, train_label
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_sample(train_data, train_label)
    return X_resampled, y_resampled


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def One_hot(labels):
    """
    :param labels:
    :return:
    """
    if len(labels.shape) == 2:
        class_list = set([i[0] for i in labels.tolist()])
    else:
        class_list = set([i for i in labels.tolist()])
    min_class = min(class_list)
    OneHot_labels = np.zeros(shape=(len(labels), len(class_list)), dtype=np.float32)
    if min_class > 0:
        new_labels = np.asarray([j - min_class for j in labels])

    else:
        new_labels = labels

    for i, vale in enumerate(new_labels):
        OneHot_labels[i, vale] = 1

    return OneHot_labels


def normalization(data):
    '''
    normalization for data
    :param data:
    :return:
    '''
    data = data.astype('float32')
    data = (data - np.min(data, axis=1, keepdims=True)) / (
        np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True) + 1e-5)
    return data


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


def read_data_datch(train_name, batch_size, train_data_shuffle=True, num_classes=18):
    train_x, train_y = [], []
    data_label = pd.read_csv(train_name, nrows=batch_size, skiprows=batch_size)
    label = data_label.values[:, -1]
    data = data_label.values[:, :-1]
    for i in range(batch_size):
        train_x.append([json.loads(i) for i in data[i]])
        train_y.append(label[i])
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_y = to_categorical(train_y, num_classes=num_classes)

    if train_data_shuffle:
        shuffle_data(train_x, train_y)

    train_x = np.reshape(train_x, [train_x.shape[0], train_x.shape[1], train_x.shape[2], 1])
    yield train_x, train_y


def get_data_for_batch(train_name, batch_size, train_data_shuffle=True, num_classes=18):
    data = read_data_datch(train_name, batch_size, train_data_shuffle=train_data_shuffle,
                           num_classes=num_classes).__next__()
    return data[0], data[1]


def load_skoda_data(calibrated_data_path=None,
                    calibrated_label_path=None,
                    raw_data_path=None,
                    raw_label_path=None,
                    split_rate=0.1,
                    train_data_shuffle=True,
                    data_path=None,
                    check_balance=True,
                    use_one_hot=True,
                    use_normalization=True,
                    expanding_data=False,
                    slid_window=True,
                    train_x_name=r'G:/data/Prepared_Data/skoda/calibrated/x_train.npy',
                    train_y_name=r'G:/data/Prepared_Data/skoda/calibrated/y_train.npy',
                    test_x_name=r'G:/data/Prepared_Data/skoda/calibrated/x_val.npy',
                    test_y_name=r'G:/data/Prepared_Data/skoda/calibrated/y_val.npy',
                    get_calibrated_data=True,
                    get_raw_data=False,
                    ws=144,
                    ss=72):
    if not os.path.exists(train_x_name) or not os.path.exists(train_y_name) or not os.path.exists(
            test_x_name) or not os.path.exists(test_y_name):
        if not os.path.exists(calibrated_data_path) or not os.path.exists(calibrated_label_path) or not os.path.exists(
                raw_data_path) or not os.path.exists(raw_label_path):
            data_dir_1 = os.path.abspath(os.path.dirname(calibrated_data_path) + os.path.sep + ".")
            data_dir_2 = os.path.abspath(os.path.dirname(calibrated_label_path) + os.path.sep + ".")
            data_dir_3 = os.path.abspath(os.path.dirname(raw_data_path) + os.path.sep + ".")
            data_dir_4 = os.path.abspath(os.path.dirname(raw_label_path) + os.path.sep + ".")
            if not os.path.exists(data_dir_1):
                os.mkdir(data_dir_1)
            if not os.path.exists(data_dir_2):
                os.mkdir(data_dir_2)
            if not os.path.exists(data_dir_3):
                os.mkdir(data_dir_3)
            if not os.path.exists(data_dir_4):
                os.mkdir(data_dir_4)
            calibrated_data, calibrated_label, raw_data, raw_label = generate_data(data_path=data_path,
                                                                                   calibrated_data_path=calibrated_data_path,
                                                                                   raw_data_path=raw_data_path,
                                                                                   calibrated_label_path=calibrated_label_path,
                                                                                   raw_label_path=raw_label_path,
                                                                                   data_shuffle=train_data_shuffle,
                                                                                   slid_window=slid_window,
                                                                                   save_files=True,
                                                                                   ws=ws,
                                                                                   ss=ss)

        else:
            calibrated_data = np.load(calibrated_data_path)
            calibrated_label = np.load(calibrated_label_path)
            raw_data = np.load(raw_data_path)
            raw_label = np.load(raw_label_path)

        calibrated_data_nums = len(calibrated_data)
        calibrated_split_point = int(calibrated_data_nums * split_rate)
        raw_data_nums = len(raw_data)
        raw_data_point = int(raw_data_nums * split_rate)

        if get_calibrated_data:
            print('Shuffle total data....')
            total_data, total_label = shuffle_data(calibrated_data, calibrated_label)
            x_train = total_data[calibrated_split_point:]
            y_train = total_label[calibrated_split_point:]
            x_val = total_data[:calibrated_split_point]
            y_val = total_label[:calibrated_split_point]

            np.save(train_x_name, x_train)
            np.save(train_y_name, y_train)
            np.save(test_x_name, x_val)
            np.save(test_y_name, y_val)

        elif get_raw_data:
            print('Shuffle total data....')
            total_data, total_label = shuffle_data(raw_data, raw_label)
            x_train = total_data[raw_data_point:]
            y_train = total_label[raw_data_point:]
            x_val = total_data[:raw_data_point]
            y_val = total_label[:raw_data_point]
            np.save(train_x_name, x_train)
            np.save(train_y_name, y_train)
            np.save(test_x_name, x_val)
            np.save(test_y_name, y_val)
        else:
            ValueError('you must choice one data in raw data and calibrated data')


    else:
        x_train = np.load(train_x_name)
        y_train = np.load(train_y_name)
        x_val = np.load(test_x_name)
        y_val = np.load(test_y_name)

    # check_balance_data(train_x=x_train,train_y=y_train)

    if check_balance:  # check data balance must before one hot encode
        check_balance_data(x_train, y_train)

    if use_one_hot:
        print('Start one hot encode.....')
        # y_train = to_categorical(y_train, num_classes=6)
        # y_val = to_categorical(y_val, num_classes=6)
        y_train = One_hot(y_train)
        y_val = One_hot(y_val)

    if use_normalization:
        print('Start normalization.....')
        x_train = normalization(x_train)
        x_val = normalization(x_val)

    if train_data_shuffle:
        print('Start shuffle for trian data.....')
        data_num = len(x_train)
        index = [i for i in range(data_num)]
        random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]

    if expanding_data:
        print('Start expanding for data.....')
        # x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
        # x_val = np.reshape(x_val, [x_val.shape[0], x_val.shape[1], x_val.shape[2], 1])
        x_train = x_train.reshape((-1, x_train.shape[1], x_train.shape[2], 1))
        x_val = x_val.reshape((-1, x_val.shape[1], x_val.shape[2], 1))

        print('expanded train_x shape is:', x_train.shape)
        print('expanded train_y shape is:', y_train.shape)
        print('expanded test_x shape is:', x_val.shape)
        print('expanded test_y shape is:', y_val.shape)

        return x_train, y_train, x_val, y_val, LABELS

    else:
        print('no expand train_x shape is:', x_train.shape)
        print('no expand train_y shape is:', y_train.shape)
        print('no expand test_x shape is:', x_val.shape)
        print('no expand test_y shape is:', y_val.shape)

    return x_train, y_train, x_val, y_val, LABELS


if __name__ == '__main__':
    get_calibrated_data = True
    get_raw_data = False
    split_rate = 0.1

    data_path = r'G:/data/skoda/right_classall_clean.mat'

    calibrated_data_path = r'G:/data/Prepared_Data/skoda/right/calibrated/x_data.npy'
    calibrated_label_path = r'G:/data/Prepared_Data/skoda/right/calibrated/y_label.npy'

    raw_data_path = r'G:/data/Prepared_Data/skoda/right/raw/x_data.npy'
    raw_label_path = r'G:/data/Prepared_Data/skoda/right/raw/y_label.npy'

    if get_calibrated_data:
        train_x_name = r'G:/data/Prepared_Data/skoda/right/calibrated/x_train.npy'
        train_y_name = r'G:/data/Prepared_Data/skoda/right/calibrated/y_train.npy'
        test_x_name = r'G:/data/Prepared_Data/skoda/right/calibrated/x_val.npy'
        test_y_name = r'G:/data/Prepared_Data/skoda/right/calibrated/y_val.npy'

    if get_raw_data:
        train_x_name = r'G:/data/Prepared_Data/skoda/right/raw/x_train.npy'
        train_y_name = r'G:/data/Prepared_Data/skoda/right/raw/y_train.npy'
        test_x_name = r'G:/data/Prepared_Data/skoda/right/raw/x_val.npy'
        test_y_name = r'G:/data/Prepared_Data/skoda/right/raw/y_val.npy'

    x_train, y_train, x_val, y_val, label_name = load_skoda_data(calibrated_data_path=calibrated_data_path,
                                                                 calibrated_label_path=calibrated_label_path,
                                                                 raw_data_path=raw_data_path,
                                                                 raw_label_path=raw_label_path,
                                                                 split_rate=split_rate,
                                                                 train_data_shuffle=True,
                                                                 data_path=data_path,
                                                                 use_one_hot=True,
                                                                 use_normalization=True,
                                                                 expanding_data=False,
                                                                 slid_window=True,
                                                                 train_x_name=train_x_name,
                                                                 train_y_name=train_y_name,
                                                                 test_x_name=test_x_name,
                                                                 test_y_name=test_y_name,
                                                                 get_calibrated_data=get_calibrated_data,
                                                                 get_raw_data=get_raw_data,
                                                                 ws=144,
                                                                 ss=72)
    print('train data shape:', x_train.shape)
    print('train label shape:', y_train.shape)
    print('val data shape:', x_val.shape)
    print('val label shape:', y_val.shape)
