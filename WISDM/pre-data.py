#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/5/09 12:23
    @author: liyakun
    The file is a new way for process wisdm data
"""
import numpy as np
import random
from imblearn.combine import SMOTEENN
import pandas as pd
import json


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


def OneHot_encode(labels):
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


def OneHot_decode(labels):
    data_nums = len(labels)
    new_labels = []
    for i in range(data_nums):
        new_labels.append(labels[i].tolist().index(1.))

    new_labels = np.asarray(new_labels)
    return new_labels


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


def check_balance_data(train_x, train_y):
    if len(train_y.shape) == 2:
        if train_y.shape[1] != 1:
            print('The label is already one hot encoded and start decode.....')
            train_y = OneHot_decode(train_y)

    assert train_x.shape[0] == train_y.shape[0]
    sample_num = len(train_y)
    class_total = []
    if len(train_y.shape) == 1:
        train_y = np.reshape(train_y, newshape=(sample_num, 1))
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


def split_train_val(data, label, split_rate, random_data=True):
    total_data_nums = len(data)
    split_point = int(total_data_nums * split_rate)
    if random_data:
        print('Shuffle train and val split.....')
        data, label = shuffle_data(data, label)

    x_train = data[split_point:]
    y_train = label[split_point:]
    x_val = data[:split_point]
    y_val = label[:split_point]
    return x_train, y_train, x_val, y_val


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

