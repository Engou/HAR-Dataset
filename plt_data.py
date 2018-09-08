#!/usr/bin/python3.5.3
# -*- coding: utf-8 -*-
""" Created on  2018/5/19 15:42
    @author: liyakun
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def OneHot_decode(labels):
    data_nums = len(labels)
    new_labels = []
    for i in range(data_nums):
        new_labels.append(labels[i].tolist().index(1.))

    new_labels = np.asarray(new_labels)
    return new_labels


def PltData(lables,
            data,
            save_path=None,
            img_title='Result'):
    if len(data.shape) == 2:
        if data.shape[1] > 1:
            data = OneHot_decode(data)

    sample_num = len(data)
    class_total = []
    if len(data.shape) == 1:
        data = np.reshape(data, newshape=(sample_num, 1))
    for i in range(sample_num):
        class_total.append(data[i, :].tolist()[0])
    class_name = sorted(list(set(class_total)))
    class_num_dic = {}
    for i in range(len(class_name)):
        class_num_dic[lables[i]] = class_total.count(class_name[i])

    lable_names, label_nums = [], []
    for name, nums in class_num_dic.items():
        lable_names.append(name)
        label_nums.append(nums)

    # print(lable_names)
    y_value = label_nums
    x_value = lable_names

    figure, ax = plt.subplots(figsize=(30, 40), dpi=80)

    plt.tick_params(labelsize=10.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 50}

    plt.title(img_title)

    plt.xlabel('Data category', font2, fontsize=30)

    plt.ylabel('Data nums', font2, fontsize=30)

    plt.xticks(range(len(x_value)), x_value)

    plt.ylim(0, 1.5 * max(y_value))

    plt.legend(prop=font1)
    # plt.xlim(0, 3.05 * max(label_nums))

    left = [i - 0.2 for i in range(len(x_value))]

    rects = plt.bar(left=left, height=y_value, width=0.3, alpha=0.8, color='blue')

    # for y, x in enumerate(label_nums):
    #     plt.text(y, x, '%s' % x)
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    # plt.show()
    if save_path is not None:
        if not os.path.exists(save_path):
            if os.path.isdir(save_path):
                os.mkdir(save_path)
                save_path = save_path + '/' + 'result_plot.png'
                plt.savefig(save_path, transprant=True)
            else:
                export_dir = os.path.abspath(os.path.dirname(save_path) + os.path.sep + ".")
                if not os.path.exists(export_dir):
                    os.mkdir(export_dir)
                plt.savefig(save_path, transprant=True)

        else:
            if os.path.isdir(save_path):
                save_path = save_path + '/' + 'result_plot.png'
                plt.savefig(save_path, transprant=True)
            else:
                plt.savefig(save_path, transprant=True)

    plt.ion()
    plt.show()
    # plt.pause(5)  # Show the result img 5s
    plt.close()
