#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/4/15 15:13
    @author: liyakun
    The file is a old way for process wisdm data
"""
import numpy as np
import os
import random
from WISDM.sliding_window import opp_sliding_window
from WISDM.user_id import user
from scipy import stats
import matplotlib.pyplot as plt

line_name = ['use_id', 'label_name', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
label_name = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']

dic = {'Walking': [],
       'Jogging': [],
       'Upstairs': [],
       'Downstairs': [],
       'Sitting': [],
       'Standing': []}

error = []

uid = user()

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

def generate_data(data_path,
                  data_shuffle=True,
                  save_file=True,
                  export_dir=None,
                  split_data=False,
                  split_rate=0.1,
                  ws=90,
                  ss=45,
                  slid_window=True):
    """
    :param data_path:
    :param data_shuffle:
    :param save_file:
    :param export_dir:
    :param split_data:
    :param split_rate:
    :param ws:
    :param ss:
    :param slid_window:
    :return:
    """

    data = open(data_path, 'r')
    lines = data.readlines()
    for k, line in enumerate(lines):
        taken = line.split(';')
        taken = taken[0].split(',')

        try:
            if taken[1] == 'Walking':
                data_label = []
                for i in taken[3:6]:
                    data_label.append(float(i))
                data_label.append(1)
                dic['Walking'].append(data_label)
            elif taken[1] == 'Jogging':
                data_label = []
                for i in taken[3:6]:
                    data_label.append(float(i))
                data_label.append(2)
                dic['Jogging'].append(data_label)
            elif taken[1] == 'Upstairs':
                data_label = []
                for i in taken[3:6]:
                    data_label.append(float(i))
                data_label.append(3)
                dic['Upstairs'].append(data_label)
            elif taken[1] == 'Downstairs':
                data_label = []
                for i in taken[3:6]:
                    data_label.append(float(i))
                data_label.append(4)
                dic['Downstairs'].append(data_label)
            elif taken[1] == 'Sitting':
                data_label = []
                for i in taken[3:6]:
                    data_label.append(float(i))
                data_label.append(5)
                dic['Sitting'].append(data_label)
            elif taken[1] == 'Standing':
                data_label = []
                for i in taken[3:6]:
                    data_label.append(float(i))
                data_label.append(6)
                dic['Standing'].append(data_label)
            else:
                pass
        except:
            error.append(k)
            continue

    print('eroor nums is:', len(error))
    Walking_data = np.asarray(dic['Walking'])
    Jogging_data = np.asarray(dic['Jogging'])
    Upstairs_data = np.asarray(dic['Upstairs'])
    Downstairs_data = np.asarray(dic['Downstairs'])
    Sitting_data = np.asarray(dic['Sitting'])
    Standing_data = np.asarray(dic['Standing'])
    total_data = np.concatenate(
        (Jogging_data, Walking_data, Upstairs_data, Downstairs_data, Sitting_data, Standing_data), axis=0)

    x_data = total_data[:, :3]

    x_data[:, 0] = feature_normalize(x_data[:, 0])
    x_data[:, 1] = feature_normalize(x_data[:, 1])
    x_data[:, 2] = feature_normalize(x_data[:, 2])

    y_data = total_data[:, 3:]
    y_data = np.reshape(y_data, (len(x_data),))
    if slid_window:
        x_data, y_data = opp_sliding_window(data_x=x_data, data_y=y_data, ws=ws, ss=ss)

        y_data = np.vstack([stats.mode(label)[0] for label in y_data])



    if data_shuffle:
        data_num = len(x_data)
        index = [i for i in range(data_num)]
        random.shuffle(index)
        x_data = x_data[index]
        y_data = y_data[index]

    # x_data = total_data[:, :3]
    # y_data = total_data[:, 3:]

    if split_data:
        total_data_nums = len(x_data)
        x_train = x_data[int(total_data_nums * split_rate):]
        y_train = y_data[int(total_data_nums * split_rate):]
        x_val = x_data[:int(total_data_nums * split_rate)]
        y_val = y_data[:int(total_data_nums * split_rate)]

        if save_file:
            if not os.path.exists(export_dir):
                os.mkdir(export_dir)
            x_train_name = os.path.join(export_dir, 'x_train.npy')
            y_train_name = os.path.join(export_dir, 'y_train.npy')
            x_val_name = os.path.join(export_dir, 'x_val.npy')
            y_val_name = os.path.join(export_dir, 'y_val.npy')

            np.save(x_train_name, x_train)
            np.save(y_train_name, y_train)
            np.save(x_val_name, x_val)
            np.save(y_val_name, y_val)
            print('Preprocess End')
            return x_train, y_train, x_val, y_val
    else:
        x_data_name = os.path.join(export_dir, 'x_data.npy')
        y_data_name = os.path.join(export_dir, 'y_data.npy')
        np.save(x_data_name, x_data)
        np.save(y_data_name, y_data)

        print('Preprocess End')
        return x_data, y_data


if __name__ == '__main__':
    data_path = r'G:/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    export_dir = r'G:/data/Prepared_Data/WISDM_ar_v1.1/data'
    x_train, y_train = generate_data(data_path, export_dir=export_dir, split_data=False, data_shuffle=False, ws=90,
                                     ss=45)
    print(x_train.shape)
    print(y_train.shape)
    # x_train = x_train.reshape(len(x_train), 1, 90, 3)
    print(x_train[:1])
    # print(x_val.shape)
    # print(y_val.shape)
    print('END')
