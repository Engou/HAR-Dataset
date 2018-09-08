#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/4/15 9:13
    @author: liyakun
    The file is a old way for process wisdm data
"""
import os
import numpy as np
import random
from WISDM.pre_wisdm_data import generate_data
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
import platform
import pandas as pd
import json

LABELS = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']

label_id = {1: 'Walking', 2: 'Jogging', 3: 'Upstairs', 4: 'Downstairs', 5: 'Sitting', 6: 'Standing'}


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
    data normalization
    :param data:
    :return:
    '''
    data = data.astype('float32')
    data = (data - np.min(data, axis=1, keepdims=True)) / (
        np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True) + 1e-5)
    return data


def check_balance_data(train_x, train_y, label_id=label_id):
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
        if label_id is None:
            print('%s rate is:%s' % (str(name), str(rate)))
        else:
            print('%s rate is:%s' % (label_id[name], str(rate)))
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


def load_WISDM_data(x_data_name,
                    y_data_name,
                    split_rate=0.1,
                    train_data_shuffle=True,
                    data_path=None,
                    use_one_hot=True,
                    use_normalization=True,
                    expanding_data=False,
                    check_balance=True,
                    export_dir=None,
                    train_x_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/x_train.npy',
                    train_y_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/y_train.npy',
                    test_x_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/x_val.npy',
                    test_y_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/y_val.npy',
                    ws=200,
                    ss=20):
    if not os.path.exists(train_x_name) or not os.path.exists(train_y_name) or not os.path.exists(
            test_x_name) or not os.path.exists(test_y_name):
        if not os.path.exists(x_data_name) or not os.path.exists(y_data_name):
            if export_dir is None:
                export_dir = os.path.abspath(os.path.dirname(x_data_name) + os.path.sep + ".")
            # print(data_path)
            x_data, y_data = generate_data(data_path=data_path,
                                           data_shuffle=train_data_shuffle,
                                           save_file=True,
                                           export_dir=export_dir,
                                           split_data=False,
                                           split_rate=split_rate,
                                           ws=ws,
                                           ss=ss,
                                           slid_window=True)

        else:
            x_data, y_data = np.load(x_data_name), np.load(y_data_name)

        total_data_nums = len(x_data)
        split_point = int(total_data_nums * split_rate)

        x_train = x_data[split_point:]
        y_train = y_data[split_point:]
        x_val = x_data[:split_point]
        y_val = y_data[:split_point]
        np.save(train_x_name, x_train)
        np.save(train_y_name, y_train)
        np.save(test_x_name, x_val)
        np.save(test_y_name, y_val)


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
        x_val = x_train.reshape((-1, x_val.shape[1], x_val.shape[2], 1))

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


def load_data(data_path,
              ws=90,
              ss=45,
              RANDOM_SEED=42,
              split_rate=0.1,
              use_normalization=True,
              train_data_shuffle=True,
              expanding_data=False,
              train_x_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/lod_data/x_train.npy',
              train_y_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/lod_data/y_train.npy',
              test_x_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/lod_data/x_val.npy',
              test_y_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/lod_data/y_val.npy'):
    if not os.path.exists(train_x_name) or not os.path.exists(train_y_name) or not os.path.exists(
            test_y_name) or not os.path.exists(test_x_name):
        colnames = ['Users', 'Activity', 'Timestamp', 'x-axis', 'y-axis', 'z-axis']
        dataset = pd.read_csv(data_path, names=colnames)
        dataset = dataset.dropna()
        segments = []
        labels = []

        for i in range(0, len(dataset) - ws, ss):
            x, y, z = [], [], []
            xs = dataset['x-axis'].values[i: i + ws]
            for j in xs:
                if isinstance(j, str):
                    j = float(j.split(';')[0])
                x.append(j)
            ys = dataset['y-axis'].values[i: i + ws]
            for j in ys:
                if isinstance(j, str):
                    j = float(j.split(';')[0])
                y.append(j)
            zs = dataset['z-axis'].values[i: i + ws]
            for j in zs:
                if isinstance(j, str):
                    j = float(j.split(';')[0])
                z.append(j)
            label = stats.mode(dataset['Activity'][i: i + ws])[0][0]
            segments.append([x, y, z])
            labels.append(label)

        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, ws, 3)
        labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

        X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=split_rate,
                                                            random_state=RANDOM_SEED)
        np.save(train_x_name, X_train)
        np.save(train_y_name, y_train)
        np.save(test_x_name, X_test)
        np.save(test_y_name, y_test)

    else:
        X_train = np.load(train_x_name)
        X_test = np.load(test_x_name)
        y_train = np.load(train_y_name)
        y_test = np.load(test_y_name)

    if use_normalization:
        print('Start normalization.....')
        X_train = normalization(X_train)
        X_test = normalization(X_test)

    if train_data_shuffle:
        print('Start shuffle for trian data.....')
        data_num = len(X_train)
        index = [i for i in range(data_num)]
        random.shuffle(index)
        X_train = X_train[index]
        y_train = y_train[index]

    if expanding_data:
        print('Start expanding for data.....')
        X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
        X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], X_test.shape[2], 1])

        print('Expanded train_x shape is:', X_train.shape)
        print('Expanded train_y shape is:', y_train.shape)
        print('Expanded test_x shape is:', X_test.shape)
        print('Expanded test_y shape is:', y_test.shape)

        return X_train, y_train, X_test, y_test, LABELS

    else:
        print('Not expanded train_x shape is:', X_train.shape)
        print('Not expanded train_y shape is:', y_train.shape)
        print('Not expanded test_x shape is:', X_test.shape)
        print('Not expanded test_y shape is:', y_test.shape)

    return X_train, y_train, X_test, y_test, LABELS


if __name__ == '__main__':
    split_rate = 0.1
    data_path = r'G:/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    x_data_name = r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/x_data.npy'
    y_data_name = r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/y_data.npy'
    train_x_name = r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/x_train.npy'
    train_y_name = r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/y_train.npy'
    test_x_name = r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/x_val.npy'
    test_y_name = r'G:/data/Prepared_Data/WISDM_ar_v1.1/data/y_val.npy'

    x_train, y_train, x_val, y_val, label_name = load_WISDM_data(x_data_name, y_data_name, split_rate=split_rate,
                                                                 use_normalization=False,
                                                                 train_data_shuffle=False, data_path=data_path, ws=90,
                                                                 ss=45)

    # print(x_train[1])
    # print('#' * 10)
    # print(x_val[1])
    #
    # x_train, y_train, x_val, y_val, label_name = load_data(data_path,
    #                                                        ws=200,
    #                                                        ss=20,
    #                                                        RANDOM_SEED=42,
    #                                                        split_rate=0.1,
    #                                                        use_normalization=False,
    #                                                        train_data_shuffle=False,
    #                                                        expanding_data=False,
    #                                                        train_x_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/lod_data/x_train.npy',
    #                                                        train_y_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/lod_data/y_train.npy',
    #                                                        test_x_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/lod_data/x_val.npy',
    #                                                        test_y_name=r'G:/data/Prepared_Data/WISDM_ar_v1.1/lod_data/y_val.npy')
    # print('train data shape:', x_train.shape)
    # print('train label shape:', y_train.shape)
    # print('val data shape:', x_val.shape)
    # print('val label shape:', y_val.shape)
    # print(x_train[1])
    # print('#' * 10)
    # print(x_val[1])
