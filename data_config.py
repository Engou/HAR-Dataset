#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/6/23 16:16
    @author: liyakun
"""
import platform
import os
from collections import OrderedDict

is_linux = (platform.system() == 'Linux')


def data_parameter_config(data_type='uci',
                          train_data_shuffle=True,
                          use_one_hot=True,
                          use_normalization=True,
                          expanding_data=True,
                          check_balance=True):
    parameter_dic = OrderedDict()
    if is_linux:
        ffp = '/home/dl/lyk/'
    else:
        ffp = 'G:/'
    if data_type.lower() == 'uci':
        window_size = 128
        sensor_nums = 9
        depth = 1
        out_channels = 6
        data_path = r'%sdata/UCI HAR Dataset/' % ffp
        export_dir = r'%sdata/Prepared_Data/UCI' % ffp
        train_x_name = r'%sdata/Prepared_Data/UCI/train_x.npy' % ffp
        train_y_name = r'%sdata/Prepared_Data/UCI/train_y.npy' % ffp
        test_x_name = r'%sdata/Prepared_Data/UCI/test_x.npy' % ffp
        test_y_name = r'%sdata/Prepared_Data/UCI/test_y.npy' % ffp

        parameter_dic['data_path'] = data_path
        parameter_dic['export_dir'] = export_dir
        parameter_dic['train_x_name'] = train_x_name
        parameter_dic['train_y_name'] = train_y_name
        parameter_dic['test_x_name'] = test_x_name
        parameter_dic['test_y_name'] = test_y_name
        parameter_dic['window_size'] = window_size
        parameter_dic['sensor_nums'] = sensor_nums
        parameter_dic['depth'] = depth
        parameter_dic['out_channels'] = out_channels
    elif data_type.lower() == 'opportunity':
        window_size = 24
        sensor_nums = 113
        depth = 1
        out_channels = 18
        balanc_data = False
        enull = False
        l = 'gestures'
        ws = 24
        ss = 24
        data_path = r'%sdata/OpportunityUCIDataset/dataset' % ffp
        export_dir = r'%sdata/prepared_data/opprtunity' % ffp
        train_x_name = r'%sdata/Prepared_Data/opprtunity/data/24_24_opprtuntiy_train_x.npy' % ffp
        train_y_name = r'%sdata/Prepared_Data/opprtunity/data/24_24_opprtuntiy_train_y.npy' % ffp
        test_x_name = r'%sdata/Prepared_Data/opprtunity/data/24_24_opprtuntiy_test_x.npy' % ffp
        test_y_name = r'%sdata/Prepared_Data/opprtunity/data/24_24_opprtuntiy_test_y.npy' % ffp
        # train_x_name = r'%sdata/Prepared_Data/opprtunity/balance_data/balance_24_24_opprtuntiy_train_x.npy' % ffp
        # train_y_name = r'%sdata/Prepared_Data/opprtunity/balance_data/balance_24_24_opprtuntiy_train_y.npy' % ffp
        # test_x_name = r'%sdata/Prepared_Data/opprtunity/balance_data/balance_24_24_opprtuntiy_test_x.npy' % ffp
        # test_y_name = r'%sdata/Prepared_Data/opprtunity/balance_data/balance_24_24_opprtuntiy_test_y.npy' % ffp

        parameter_dic['l'] = l
        parameter_dic['ws'] = ws
        parameter_dic['ss'] = ss
        parameter_dic['data_path'] = data_path
        parameter_dic['export_dir'] = export_dir
        parameter_dic['train_x_name'] = train_x_name
        parameter_dic['train_y_name'] = train_y_name
        parameter_dic['test_x_name'] = test_x_name
        parameter_dic['test_y_name'] = test_y_name
        parameter_dic['window_size'] = window_size
        parameter_dic['sensor_nums'] = sensor_nums
        parameter_dic['depth'] = depth
        parameter_dic['out_channels'] = out_channels
        parameter_dic['balanc_data'] = balanc_data
        parameter_dic['enull'] = enull

    elif data_type.lower() == 'opportunity_enull':
        window_size = 24
        sensor_nums = 113
        depth = 1
        out_channels = 18
        balanc_data = False
        enull = True
        l = 'gestures'
        ws = 24
        ss = 24

        data_path = r'%sdata/OpportunityUCIDataset/dataset' % ffp
        export_dir = r'%sdata/prepared_data/opprtunity' % ffp
        train_x_name = r'%sdata/Prepared_Data/opprtunity/enull_data/24_24_opprtuntiy_train_x.npy' % ffp
        train_y_name = r'%sdata/Prepared_Data/opprtunity/enull_data/24_24_opprtuntiy_train_y.npy' % ffp
        test_x_name = r'%sdata/Prepared_Data/opprtunity/enull_data/24_24_opprtuntiy_test_x.npy' % ffp
        test_y_name = r'%sdata/Prepared_Data/opprtunity/enull_data/24_24_opprtuntiy_test_y.npy' % ffp

        parameter_dic['l'] = l
        parameter_dic['ws'] = ws
        parameter_dic['ss'] = ss
        parameter_dic['data_path'] = data_path
        parameter_dic['export_dir'] = export_dir
        parameter_dic['train_x_name'] = train_x_name
        parameter_dic['train_y_name'] = train_y_name
        parameter_dic['test_x_name'] = test_x_name
        parameter_dic['test_y_name'] = test_y_name
        parameter_dic['window_size'] = window_size
        parameter_dic['sensor_nums'] = sensor_nums
        parameter_dic['depth'] = depth
        parameter_dic['out_channels'] = out_channels
        parameter_dic['balanc_data'] = balanc_data
        parameter_dic['enull'] = enull

    elif data_type.lower() == 'opportunity_balance':
        window_size = 24
        sensor_nums = 113
        depth = 1
        out_channels = 18
        balanc_data = True
        enull = False
        l = 'gestures'
        ws = 24
        ss = 24
        data_path = r'%sdata/OpportunityUCIDataset/dataset' % ffp
        export_dir = r'%sdata/prepared_data/opprtunity' % ffp
        # train_x_name = r'%sdata/Prepared_Data/opprtunity/data/24_24_opprtuntiy_train_x.npy' % ffp
        # train_y_name = r'%sdata/Prepared_Data/opprtunity/data/24_24_opprtuntiy_train_y.npy' % ffp
        # test_x_name = r'%sdata/Prepared_Data/opprtunity/data/24_24_opprtuntiy_test_x.npy' % ffp
        # test_y_name = r'%sdata/Prepared_Data/opprtunity/data/24_24_opprtuntiy_test_y.npy' % ffp
        train_x_name = r'%sdata/Prepared_Data/opprtunity/balance_data/balance_24_24_opprtuntiy_train_x.npy' % ffp
        train_y_name = r'%sdata/Prepared_Data/opprtunity/balance_data/balance_24_24_opprtuntiy_train_y.npy' % ffp
        test_x_name = r'%sdata/Prepared_Data/opprtunity/balance_data/balance_24_24_opprtuntiy_test_x.npy' % ffp
        test_y_name = r'%sdata/Prepared_Data/opprtunity/balance_data/balance_24_24_opprtuntiy_test_y.npy' % ffp

        parameter_dic['l'] = l
        parameter_dic['ws'] = ws
        parameter_dic['ss'] = ss
        parameter_dic['data_path'] = data_path
        parameter_dic['export_dir'] = export_dir
        parameter_dic['train_x_name'] = train_x_name
        parameter_dic['train_y_name'] = train_y_name
        parameter_dic['test_x_name'] = test_x_name
        parameter_dic['test_y_name'] = test_y_name
        parameter_dic['window_size'] = window_size
        parameter_dic['sensor_nums'] = sensor_nums
        parameter_dic['depth'] = depth
        parameter_dic['out_channels'] = out_channels
        parameter_dic['balanc_data'] = balanc_data
        parameter_dic['enull'] = enull

    elif data_type.lower() == 'wisdm_90_45':
        split_rate = 0.2
        ws = 90
        ss = 45
        window_size = 90
        sensor_nums = 3
        depth = 1
        out_channels = 6

        data_path = r'%sdata/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt' % ffp
        export_dir = r'%sdata/Prepared_Data/WISDM_ar_v1.1/data' % ffp
        x_data_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/90_45_data/x_data.npy' % ffp
        y_data_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/90_45_data/y_data.npy' % ffp
        train_x_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/90_45_data/x_train.npy' % ffp
        train_y_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/90_45_data/y_train.npy' % ffp
        test_x_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/90_45_data/x_val.npy' % ffp
        test_y_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/90_45_data/y_val.npy' % ffp

        parameter_dic['split_rate'] = split_rate
        parameter_dic['ws'] = ws
        parameter_dic['ss'] = ss
        parameter_dic['data_path'] = data_path
        parameter_dic['export_dir'] = export_dir
        parameter_dic['x_data_name'] = x_data_name
        parameter_dic['y_data_name'] = y_data_name
        parameter_dic['train_x_name'] = train_x_name
        parameter_dic['train_y_name'] = train_y_name
        parameter_dic['test_x_name'] = test_x_name
        parameter_dic['test_y_name'] = test_y_name
        parameter_dic['window_size'] = window_size
        parameter_dic['sensor_nums'] = sensor_nums
        parameter_dic['depth'] = depth
        parameter_dic['out_channels'] = out_channels

    elif data_type.lower() == 'wisdm_200_20':
        split_rate = 0.2
        ws = 200
        ss = 20
        window_size = 200
        sensor_nums = 3
        depth = 1
        out_channels = 6

        data_path = r'%sdata/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt' % ffp
        export_dir = r'%sdata/Prepared_Data/WISDM_ar_v1.1/data' % ffp
        x_data_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/200_20_data/x_data.npy' % ffp
        y_data_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/200_20_data/y_data.npy' % ffp
        train_x_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/200_20_data/x_train.npy' % ffp
        train_y_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/200_20_data/y_train.npy' % ffp
        test_x_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/200_20_data/x_val.npy' % ffp
        test_y_name = r'%sdata/Prepared_Data/WISDM_ar_v1.1/200_20_data/y_val.npy' % ffp

        parameter_dic['split_rate'] = split_rate
        parameter_dic['ws'] = ws
        parameter_dic['ss'] = ss
        parameter_dic['data_path'] = data_path
        parameter_dic['export_dir'] = export_dir
        parameter_dic['x_data_name'] = x_data_name
        parameter_dic['y_data_name'] = y_data_name
        parameter_dic['train_x_name'] = train_x_name
        parameter_dic['train_y_name'] = train_y_name
        parameter_dic['test_x_name'] = test_x_name
        parameter_dic['test_y_name'] = test_y_name
        parameter_dic['window_size'] = window_size
        parameter_dic['sensor_nums'] = sensor_nums
        parameter_dic['depth'] = depth
        parameter_dic['out_channels'] = out_channels

    elif data_type.lower() == 'skoda':
        type = 'right'
        window_size = 144
        sensor_nums = 30
        depth = 1
        out_channels = 11
        ws = 144
        ss = 72
        split_rate = 0.2
        get_calibrated_data = True
        get_raw_data = False
        slid_window = True

        data_path = r'%sdata/skoda/right_classall_clean.mat' % ffp

        calibrated_data_path = r'%sdata/Prepared_Data/skoda/%s/calibrated/x_data.npy' % (ffp, type)
        calibrated_label_path = r'%sdata/Prepared_Data/skoda/%s/calibrated/y_label.npy' % (ffp, type)

        raw_data_path = r'%sdata/Prepared_Data/skoda/%s/raw/x_data.npy' % (ffp, type)
        raw_label_path = r'%sdata/Prepared_Data/skoda/%s/raw/y_label.npy' % (ffp, type)

        if get_calibrated_data:
            train_x_name = r'%sdata/Prepared_Data/skoda/%s/calibrated/x_train.npy' % (ffp, type)
            train_y_name = r'%sdata/Prepared_Data/skoda/%s/calibrated/y_train.npy' % (ffp, type)
            test_x_name = r'%sdata/Prepared_Data/skoda/%s/calibrated/x_val.npy' % (ffp, type)
            test_y_name = r'%sdata/Prepared_Data/skoda/%s/calibrated/y_val.npy' % (ffp, type)

        if get_raw_data:
            train_x_name = r'%sdata/Prepared_Data/skoda/%s/raw/x_train.npy' % (ffp, type)
            train_y_name = r'%sdata/Prepared_Data/skoda/%s/raw/y_train.npy' % (ffp, type)
            test_x_name = r'%sdata/Prepared_Data/skoda/%s/raw/x_val.npy' % (ffp, type)
            test_y_name = r'%sdata/Prepared_Data/skoda/%s/raw/y_val.npy' % (ffp, type)

        else:
            ValueError('you must choice one data in raw data and calibrated data')

        parameter_dic['split_rate'] = split_rate
        parameter_dic['ws'] = ws
        parameter_dic['ss'] = ss
        parameter_dic['data_path'] = data_path
        parameter_dic['calibrated_data_path'] = calibrated_data_path
        parameter_dic['calibrated_label_path'] = calibrated_label_path
        parameter_dic['raw_data_path'] = raw_data_path
        parameter_dic['raw_label_path'] = raw_label_path
        parameter_dic['train_x_name'] = train_x_name
        parameter_dic['train_y_name'] = train_y_name
        parameter_dic['test_x_name'] = test_x_name
        parameter_dic['test_y_name'] = test_y_name
        parameter_dic['get_calibrated_data'] = get_calibrated_data
        parameter_dic['get_raw_data'] = get_raw_data
        parameter_dic['slid_window'] = slid_window
        parameter_dic['window_size'] = window_size
        parameter_dic['sensor_nums'] = sensor_nums
        parameter_dic['depth'] = depth
        parameter_dic['out_channels'] = out_channels

    else:
        ValueError('data type is error,it should in ‘uci’,‘opportunity’,‘wisdm’,‘skoda’')

    parameter_dic['train_data_shuffle'] = train_data_shuffle
    parameter_dic['use_one_hot'] = use_one_hot
    parameter_dic['use_normalization'] = use_normalization
    parameter_dic['expanding_data'] = expanding_data
    parameter_dic['check_balance'] = check_balance
    return parameter_dic
