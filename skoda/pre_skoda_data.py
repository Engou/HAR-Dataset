#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/4/23 11:13
    @author: liyakun
    The file is a old way for process wisdm data
"""
import os
import random
from collections import Counter

import numpy
import numpy as np
from scipy import stats
from scipy.io import loadmat
from sliding_window import opp_sliding_window
import copy
from numpy.lib.stride_tricks import as_strided as ast

skoda_libale_dict = {32: 'null class', 48: 'write on notepad', 49: 'open hood', 50: 'close hood',
                     51: 'check graps on the front door', 52: 'open left front door', 53: 'close left front door',
                     54: 'close booth left door', 55: 'check trunk gaps', 56: 'open and close trunk',
                     57: 'check steering wheel'}


def generate_data(data_path,
                  calibrated_data_path=None,
                  raw_data_path=None,
                  calibrated_label_path=None,
                  raw_label_path=None,
                  data_shuffle=True,
                  slid_window=True,
                  save_files=True,
                  ws=144,
                  ss=72):
    '''
    Matrix format: one line per sample.
    Column 1:label
    Column 2+s*7: sensor id
    Column 2+s*7+1: X acceleration calibrated
    Column 2+s*7+2: Y acceleration calibrated
    Column 2+s*7+3: Z acceleration calibrated
    Column 2+s*7+4: X acceleration raw
    Column 2+s*7+5: Y acceleration raw
    Column 2+s*7+6: Z acceleration raw
    with s=0...29 are the sensor axis (10 3­axis sensors = 10x3 = 30 axis). Sensor node number is mod(s,3).
    Calibrated acceleration means acceleration in milli­g units (1000 = earth gravity vector).
    Raw acceleration is ADC readout.
    :param data_path:
    :return: return four data,the first data is calibrated data,the second data is calibrated label,the third data is raw data and the fourth data is raw label
    '''
    if not os.path.exists(calibrated_data_path) or not os.path.exists(raw_data_path) or not os.path.exists(
            calibrated_label_path) or not os.path.exists(raw_label_path):
        mat_data = loadmat(data_path)
        for value in mat_data.values():
            if isinstance(value, numpy.ndarray):
                tmp_data = value
                print('original data shape is: ', tmp_data.shape)
        calibrated_data = numpy.hstack([tmp_data[:, 1 + idx * 7 + 1: 1 + idx * 7 + 4] for idx in range(10)]).astype(
            'float32')
        raw_data = numpy.hstack([tmp_data[:, 1 + idx * 7 + 4: 1 + idx * 7 + 7] for idx in range(10)]).astype('float32')

        label_array = tmp_data[:, 0]
        label_idx_dict = dict(zip(numpy.unique(label_array), numpy.arange(len(numpy.unique(label_array)))))
        print('label_dict: ', label_idx_dict)

        calibrated_label = np.array([label_idx_dict[label] for label in label_array])
        # calibrated_label = np.reshape(calibrated_label, newshape=(len(calibrated_label), 1))
        raw_label = copy.copy(calibrated_label)

        if slid_window:
            calibrated_data, calibrated_label = opp_sliding_window(data_x=calibrated_data, data_y=calibrated_label,
                                                                   ws=ws, ss=ss)
            raw_data, raw_label = opp_sliding_window(data_x=raw_data, data_y=raw_label, ws=ws, ss=ss)

            calibrated_label = np.vstack([stats.mode(label)[0] for label in calibrated_label])
            raw_label = np.vstack([stats.mode(label)[0] for label in raw_label])

        if data_shuffle:
            data_num = len(calibrated_label)
            index = [i for i in range(data_num)]
            random.shuffle(index)
            calibrated_data = calibrated_data[index]
            calibrated_label = calibrated_label[index]

            raw_data = raw_data[index]
            raw_label = raw_label[index]

        if save_files:
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
            np.save(calibrated_data_path, calibrated_data)
            np.save(calibrated_label_path, calibrated_label)
            np.save(raw_data_path, raw_data)
            np.save(raw_label_path, raw_label)


    else:
        calibrated_data = np.load(calibrated_data_path)
        calibrated_label = np.load(calibrated_label_path)
        raw_data = np.load(raw_data_path)
        raw_label = np.load(raw_label_path)

    print('Preprocess End')

    print('total calibrated data shape: ', calibrated_data.shape)
    print('total raw_data data shape: ', raw_data.shape)
    print('total calibrated label shape:', calibrated_label.shape)
    print('total raw label shape:', raw_label.shape)

    return calibrated_data, calibrated_label, raw_data, raw_label


    # return calibrated_data, raw_data, new_label


if __name__ == '__main__':
    data_path = r'G:/data/skoda/right_classall_clean.mat'

    calibrated_data_path = r'G:/data/Prepared_Data/skoda/right/calibrated/x_data.npy'
    raw_data_path = r'G:/data/Prepared_Data/skoda/right/raw/x_data.npy'
    calibrated_label_path = r'G:/data/Prepared_Data/skoda/right/calibrated/y_label.npy'
    raw_label_path = r'G:/data/Prepared_Data/skoda/right/raw/y_label.npy'

    generate_data(data_path, calibrated_data_path=calibrated_data_path, calibrated_label_path=calibrated_label_path,
                  raw_data_path=raw_data_path, raw_label_path=raw_label_path)
