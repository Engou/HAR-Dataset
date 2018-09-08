#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/5/19 10:26
    @author: liyakun
"""
from data_config import data_parameter_config
from plt_data import PltData
import platform
import sys
import os
import numpy as np

is_linux = (platform.system() == 'Linux')

if is_linux:
    data_list = [r'/home/dl/lyk/data/data_prepare/opprtunity', r'/home/dl/lyk/data/data_prepare/UCI',
                 r'/home/dl/lyk/data/data_prepare/WISDM', 'G:/data/data_prepare/skoda']
    sys.path.extend(data_list)
    from get_oppr_data import load_oppr_data, get_data_for_batch
    from get_UCI_data import load_UCI_data
    from get_wisdm__data import load_WISDM_data
    from get_skoda__data import load_skoda_data

    # sess = get_session()
else:
    data_list = [r'G:/data/data_prepare/opprtunity', r'G:/data/data_prepare/UCI',
                 'G:/data/data_prepare/WISDM', 'G:/data/data_prepare/skoda']
    sys.path.extend(data_list)
    from get_oppr_data import load_oppr_data, get_data_for_batch
    from get_UCI_data import load_UCI_data
    from get_WISDM_data import load_wisdm_data
    from get_skoda__data import load_skoda_data

data_type = 'uci'
# data_type = 'opportunity'
# data_type = 'wisdm'
# data_type = 'skoda'

parameter_dic = data_parameter_config(data_type=data_type)


def get_data(data_type='UCI',
             train_data_shuffle=True,
             use_one_hot=True,
             use_normalization=True,
             expanding_data=True,
             check_balance=True,
             plt_data=True,
             plt_path=None,
             plt_title=None):
    data_parameter_dic = data_parameter_config(data_type,
                                               train_data_shuffle=train_data_shuffle,
                                               use_one_hot=use_one_hot,
                                               use_normalization=use_normalization,
                                               expanding_data=expanding_data,
                                               check_balance=check_balance)
    # print(data_parameter_dic)
    # print(data_parameter_dic)
    data_type = data_type.lower().split('_')[0]
    if data_type.lower() == 'uci':
        print('Load The UCI Data')
        X_train, y_train, X_test, y_test, labels = load_UCI_data(data_path=data_parameter_dic['data_path'],
                                                                 export_dir=data_parameter_dic['export_dir'],
                                                                 train_x_name=data_parameter_dic['train_x_name'],
                                                                 train_y_name=data_parameter_dic['train_y_name'],
                                                                 test_x_name=data_parameter_dic['test_x_name'],
                                                                 test_y_name=data_parameter_dic['test_y_name'],
                                                                 train_data_shuffle=data_parameter_dic[
                                                                     'train_data_shuffle'],
                                                                 use_one_hot=data_parameter_dic['use_one_hot'],
                                                                 use_normalization=data_parameter_dic[
                                                                     'use_normalization'],
                                                                 expanding_data=data_parameter_dic['expanding_data'],
                                                                 check_balance=data_parameter_dic['check_balance'])

    elif data_type.lower() == 'opportunity':
        print('Load The OpportunityUCI Data')
        X_train, y_train, X_test, y_test, labels = load_oppr_data(data_path=data_parameter_dic['data_path'],
                                                                  export_dir=data_parameter_dic['export_dir'],
                                                                  l=data_parameter_dic['l'],
                                                                  ws=data_parameter_dic['ws'],
                                                                  ss=data_parameter_dic['ss'],
                                                                  train_data_shuffle=data_parameter_dic[
                                                                      'train_data_shuffle'],
                                                                  expanding_data=data_parameter_dic['expanding_data'],
                                                                  use_one_hot=data_parameter_dic['use_one_hot'],
                                                                  use_normalization=data_parameter_dic[
                                                                      'use_normalization'],
                                                                  train_x_name=data_parameter_dic['train_x_name'],
                                                                  train_y_name=data_parameter_dic['train_y_name'],
                                                                  test_x_name=data_parameter_dic['test_x_name'],
                                                                  test_y_name=data_parameter_dic['test_y_name'],
                                                                  balanc_data=data_parameter_dic['balanc_data'],
                                                                  enull=data_parameter_dic['enull'])
    elif data_type.lower() == 'wisdm':
        print('Load The WISDM Data')
        X_train, y_train, X_test, y_test, labels = load_wisdm_data(split_rate=data_parameter_dic['split_rate'],
                                                                   data_path=data_parameter_dic['data_path'],
                                                                   train_data_shuffle=data_parameter_dic[
                                                                       'train_data_shuffle'],
                                                                   expanding_data=data_parameter_dic['expanding_data'],
                                                                   use_one_hot=data_parameter_dic['use_one_hot'],
                                                                   use_normalization=data_parameter_dic[
                                                                       'use_normalization'],
                                                                   check_balance=data_parameter_dic['check_balance'],
                                                                   train_x_name=data_parameter_dic['train_x_name'],
                                                                   train_y_name=data_parameter_dic['train_y_name'],
                                                                   test_x_name=data_parameter_dic['test_x_name'],
                                                                   test_y_name=data_parameter_dic['test_y_name'],
                                                                   ws=data_parameter_dic['ws'],
                                                                   ss=data_parameter_dic['ss'])

    elif data_type.lower() == 'skoda':
        print(data_parameter_dic['out_channels'])
        print('Load The skoda data')
        X_train, y_train, X_test, y_test, labels = load_skoda_data(
            calibrated_data_path=data_parameter_dic['calibrated_data_path'],
            calibrated_label_path=data_parameter_dic['calibrated_label_path'],
            raw_data_path=data_parameter_dic['raw_data_path'],
            raw_label_path=data_parameter_dic['raw_label_path'],
            train_data_shuffle=data_parameter_dic['train_data_shuffle'],
            data_path=data_parameter_dic['data_path'],
            check_balance=data_parameter_dic['check_balance'],
            use_one_hot=data_parameter_dic['use_one_hot'],
            use_normalization=data_parameter_dic['use_normalization'],
            expanding_data=data_parameter_dic['expanding_data'],
            slid_window=data_parameter_dic['slid_window'],
            train_x_name=data_parameter_dic['train_x_name'],
            train_y_name=data_parameter_dic['train_y_name'],
            test_x_name=data_parameter_dic['test_x_name'],
            test_y_name=data_parameter_dic['test_y_name'],
            get_calibrated_data=data_parameter_dic['get_calibrated_data'],
            get_raw_data=data_parameter_dic['get_raw_data'],
            ws=data_parameter_dic['ws'],
            ss=data_parameter_dic['ss'],
            split_rate=data_parameter_dic['split_rate'])
    else:
        ValueError('The data type is error')

    if plt_data:
        data = np.concatenate((y_train, y_test), axis=0)
    export_dir = os.path.abspath(os.path.dirname(data_parameter_dic['train_x_name']) + os.path.sep + ".")
    save_path = export_dir + '/' + 'total_data_distributed.png'
    PltData(lables=labels, data=data, save_path=save_path)

    return X_train, y_train, X_test, y_test, labels


if __name__ == '__main__':
    from data_config import data_parameter_config

    data_type = ['UCI', 'Opportunity', 'opportunity_balance', 'opportunity_enull', 'skoda', 'wisdm_200_20',
                 'wisdm_90_45']
    # parameter_dic = data_parameter_config(data_type)
    x_train, y_train, x_val, y_val, labels = get_data(data_type='wisdm_200_20', plt_data=True)

    print('train data shape:', x_train.shape)
    print('train label shape:', y_train.shape)
    print('val data shape:', x_val.shape)
    print('val label shape:', y_val.shape)
    print(labels)
    # from plt_data import PltData
    #
    # PltData(lables=labels,data=y_train)
