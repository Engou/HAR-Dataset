#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/3/28 20:39
    @author: liyakun
"""
from __future__ import division
import os
import numpy as np
from preprocess_data import generate_data
import random
from imblearn.combine import SMOTEENN
import platform
import pandas as pd
import json

is_linux = (platform.system() == 'Linux')


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


def normalization(data):
    '''
    对数据进行归一化
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
    index = [i for i in range(data_num)]
    random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]
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


def load_oppr_data(data_path=r'G:/data/OpportunityUCIDataset/dataset',
                   export_dir=r'G:/data/prepared_data/opprtunity',
                   l='gestures',
                   ws=24,
                   ss=24,
                   train_data_shuffle=True,
                   expanding_data=True,
                   balanc_data=False,
                   use_one_hot=True,
                   use_normalization=True,
                   enull=False,
                   train_x_name=r'G:/data/prepared_data/opprtunity/24_24_opprtuntiy_train_x.npy',
                   train_y_name=r'G:/data/prepared_data/opprtunity/24_24_opprtuntiy_train_y.npy',
                   test_x_name=r'G:/data/prepared_data/opprtunity/24_24_opprtuntiy_test_x.npy',
                   test_y_name=r'G:/data/prepared_data/opprtunity/24_24_opprtuntiy_test_y.npy'):
    LABELS = ['No-Movement',
              'pen-Door-1',
              'Open-Door-2',
              'Close-Door-1',
              'Close-Door-2',
              'Open-Fridge',
              'lose-Fridge',
              'Open-Dishwasher',
              'Close-Dishwasher',
              'Open-Drawer-1',
              'Close-Drawer-1',
              'Open-Drawer-2',
              'Close-Drawer-2',
              'Open-Drawer-3',
              'Close-Drawer-3',
              'Clean-Table',
              'Drink-from-Cup',
              'Toggle-Switch']
    if enull:
        LABELS.pop(0)

    print('Strart reload the data......')
    if not os.path.exists(train_x_name) or not os.path.exists(train_y_name) or not os.path.exists(
            test_x_name) or not os.path.exists(test_y_name):
        if export_dir is None:
            export_dir = os.path.abspath(os.path.dirname(train_x_name) + os.path.sep + ".")
        train_x, train_y, test_x, test_y = generate_data(dataset_dir=data_path,
                                                         target_filename=export_dir,
                                                         label=l,
                                                         ws=ws,
                                                         ss=ss,
                                                         balanc_data=balanc_data)
    else:
        train_x = np.load(train_x_name)
        train_y = np.load(train_y_name)
        test_x = np.load(test_x_name)
        test_y = np.load(test_y_name)

    if enull:
        train_total_nums = len(train_y)
        test_total_nums = len(test_y)
        train_null_index = [i for i in range(train_total_nums) if train_y[i][0] == 0]
        test_null_index = [i for i in range(test_total_nums) if test_y[i][0] == 0]
        train_x = np.delete(train_x, train_null_index, 0)
        train_y = np.delete(train_y, train_null_index, 0)
        test_x = np.delete(test_x, test_null_index, 0)
        test_y = np.delete(test_y, test_null_index, 0)

        np.save(train_x_name, train_x)
        np.save(train_y_name, train_y)
        np.save(test_x_name, test_x)
        np.save(test_y_name, test_y)

    print('Start check data balance.....')
    check_balance_data(train_x, train_y)

    if use_one_hot:
        print('Start one hot encode.....')
        train_y = OneHot_encode(train_y)
        test_y = OneHot_encode(test_y)

    if use_normalization:
        print('Start normalization.....')
        train_x = normalization(train_x)
        test_x = normalization(test_x)

    if train_data_shuffle:
        print('Start shuffle for trian data.....')
        data_num = len(train_x)
        index = [i for i in range(data_num)]
        random.shuffle(index)
        train_x = train_x[index]
        train_y = train_y[index]

    if expanding_data:
        print('Start expanding for data.....')
        train_x = np.reshape(train_x, [train_x.shape[0], train_x.shape[1], train_x.shape[2], 1])
        test_x = np.reshape(test_x, [test_x.shape[0], test_x.shape[1], test_x.shape[2], 1])
        # x_train = train_x.reshape((-1, 1, train_x.shape[1], train_x.shape[2]))
        # test_x = x_train.reshape((-1, 1, test_x.shape[1], test_x.shape[2]))

        print('expanded train_x shape is:', train_x.shape)
        print('expanded train_y shape is:', train_y.shape)
        print('expanded test_x shape is:', test_x.shape)
        print('expanded test_y shape is:', test_y.shape)

        return train_x, train_y, test_x, test_y, LABELS

    else:
        print('no expand train_x shape is:', train_x.shape)
        print('no expand train_y shape is:', train_y.shape)
        print('no expand test_x shape is:', test_x.shape)
        print('no expand test_y shape is:', test_y.shape)

        return train_x, train_y, test_x, test_y, LABELS


if __name__ == '__main__':
    if is_linux:
        data_path = r'/home/lyk/data/OpportunityUCIDataset/dataset'
        export_dir = r'home/lyk/data/prepared_data/opprtunity'
        train_x_name = r'home/lyk/data/prepared_data/opprtunity/24_24_opprtuntiy_train_x.npy'
        train_y_name = r'home/lyk/data/prepared_data/opprtunity/24_24_opprtuntiy_train_y.npy'
        test_x_name = r'home/lyk/data/prepared_data/opprtunity/24_24_opprtuntiy_test_x.npy'
        test_y_name = r'home/lyk/data/prepared_data/opprtunity/24_24_opprtuntiy_test_y.npy'
        train_name = r'home/lyk/data/prepared_data/opprtunity/24_24_opprtuntiy_test.csv'
        test_name = r'home/lyk/data/prepared_data/opprtunity/24_24_opprtuntiy_train.csv'
    else:
        data_path = r'G:/data/OpportunityUCIDataset/dataset'
        export_dir = r'G:/data/prepared_data/opprtunity'
        train_x_name = r'G:/data/Prepared_Data/opprtunity/data/24_24_opprtuntiy_train_x.npy'
        train_y_name = r'G:/data/Prepared_Data/opprtunity/data/24_24_opprtuntiy_train_y.npy'
        test_x_name = r'G:/data/Prepared_Data/opprtunity/data/24_24_opprtuntiy_test_x.npy'
        test_y_name = r'G:/data/Prepared_Data/opprtunity/data/24_24_opprtuntiy_test_y.npy'
        train_name = r'G:/data/Prepared_Data/opprtunity/data/24_24_opprtuntiy_test.csv'
        test_name = r'G:/data/Prepared_Data/opprtunity/data/24_24_opprtuntiy_train.csv'

    train_x, train_y, test_x, test_y, labels = load_oppr_data(data_path=data_path,
                                                              export_dir=export_dir,
                                                              l='gestures',
                                                              ws=24,
                                                              ss=24,
                                                              train_data_shuffle=True,
                                                              expanding_data=True,
                                                              balanc_data=False,
                                                              use_one_hot=True,
                                                              use_normalization=True,
                                                              train_x_name=train_x_name,
                                                              train_y_name=train_y_name,
                                                              test_x_name=test_x_name,
                                                              test_y_name=test_y_name)

    print('End')
