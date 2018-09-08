# coding:utf-8
'''
    When this data set is not balanced,Solve the balance problem using the somte algorithm for train data,
    However, this algorithm requires a large memory space(32GB~~),And the processing time is longer(4 to 6 hours or more),
    So,If the memory is small, please process the data in segments
    This program will generate 10 files
'''
from random import shuffle
import os
import zipfile
from io import BytesIO
# import cPickle as cp
from scipy import stats
import numpy as np
from pandas import Series
from imblearn.combine import SMOTEENN
from sliding_window import opp_sliding_window
import platform
import pandas as pd

is_linux = (platform.system() == 'Linux')

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

# Hardcoded names of the files defining the OPPORTUNITY challenge data. As named in the original data.
OPPORTUNITY_DATA_FILES = ['S1-Drill.dat',
                          'S1-ADL1.dat',
                          'S1-ADL2.dat',
                          'S1-ADL3.dat',
                          'S1-ADL4.dat',
                          'S1-ADL5.dat',
                          'S2-Drill.dat',
                          'S2-ADL1.dat',
                          'S2-ADL2.dat',
                          'S2-ADL3.dat',
                          'S2-ADL4.dat',
                          'S2-ADL5.dat',
                          'S3-Drill.dat',
                          'S3-ADL1.dat',
                          'S3-ADL2.dat',
                          'S3-ADL3.dat',
                          'S3-ADL4.dat',
                          'S3-ADL5.dat',
                          'S4-Drill.dat',
                          'S4-ADL1.dat',
                          'S4-ADL2.dat',
                          'S4-ADL3.dat',
                          'S4-ADL4.dat',
                          'S4-ADL5.dat']

# Hardcoded thresholds to define global maximums and minimums for every one of the 113 sensor channels employed in the
# OPPORTUNITY challenge
NORM_MAX_THRESHOLDS = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                       3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                       3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                       3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                       3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                       3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                       3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                       3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                       3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                       250, 25, 200, 5000, 5000, 5000, 5000, 5000, 5000,
                       10000, 10000, 10000, 10000, 10000, 10000, 250, 250, 25,
                       200, 5000, 5000, 5000, 5000, 5000, 5000, 10000, 10000,
                       10000, 10000, 10000, 10000, 250, ]

NORM_MIN_THRESHOLDS = [-3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                       -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                       -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                       -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                       -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                       -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                       -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                       -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                       -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                       -250, -100, -200, -5000, -5000, -5000, -5000, -5000, -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250, -250, -100,
                       -200, -5000, -5000, -5000, -5000, -5000, -5000, -10000, -10000,
                       -10000, -10000, -10000, -10000, -250, ]


def select_columns_opp(data):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #                     included-excluded
    features_delete = np.arange(46, 50)  # 四元数(4)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])  # 四元数(4)
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])  # 四元数(4)
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])  # 四元数(4)
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])  # 四元数(4)
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])  # 物体上的传感器(109)
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])  # 标签(5)
    return np.delete(data, features_delete, 1)  # 1时间戳+113数据+2标签(244,250)(250-4*5-109-5=116)


def normalize(data, max_list, min_list):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 113 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 113 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i] - min_list[i]) / diffs[i]
    # Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def divide_x_y(data, label):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, 1:114]
    if label not in ['locomotion', 'gestures']:
        raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, 114]  # Locomotion label

    elif label == 'gestures':
        data_y = data[:, 115]  # Gestures label
    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data = select_columns_opp(data)

    # Colums are segmentd into features and labels
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in
                       data_x.T]).T  # If you do not add T, the horizontal difference, that is, according to the data in one line, add nan in a row, add T, then interpolate in the column direction, such as [:, 34:36], the original data is nan (except 50671 0), interpolated by column, then [:50671,34:36] are both nan, and [50671:,34:36] is 0. Where [:, 34:36 are accelerometers in the right hand]

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0
    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)
    return data_x, data_y


def check_balance_data(train_x, train_y):
    """
    :param train_x:the data_x
    :param train_y: the data_y
    :return: return a dict {label:rate}
    """
    assert train_x.shape[0] == train_y.shape[0]
    sample_num = len(train_y)
    # class_total = []
    # print(train_y.tolist())
    # for i in range(sample_num):
    #     class_total.append(train_y.tolist()[i])
    label = train_y.tolist()
    class_total = [label[i] for i in range(sample_num)]
    # for i in range(sample_num):
    #     class_total.append(train_y[i,:].tolist()[0])
    class_name = list(set(class_total))
    class_num_dic = {}
    for i in class_name:
        class_num_dic[i] = float(class_total.count(i)) / float(sample_num)
    for name, rate in class_num_dic.items():
        print('%s rate is:%s' % (str(name), str(rate)))
    print('the sample total num is:%d' % sample_num)
    return class_num_dic


def balance_data(train_x, trian_y):
    sm = SMOTEENN()
    train_x, train_y = sm.fit_sample(train_x, trian_y)
    return train_x, train_y


def generate_data(dataset_dir, target_filename, label, ws, ss, balance_rate=0.4, balanc_data=False):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """

    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    print('Processing dataset files ...')
    for filename in OPPORTUNITY_DATA_FILES:
        file_path = os.path.join(dataset_dir, filename)
        try:
            print('... file {0}'.format(file_path))
            data = np.loadtxt(file_path)
            x, y = process_dataset_file(data, label)
            # x, y = opp_sliding_window(x, y, ws, ss)
            if 'S4' in filename:
                test_x.append(x)
                test_y.append(y)
            else:
                train_x.append(x)
                train_y.append(y)
        except KeyError:
            print('ERROR: Did not find {0}'.format(file_path))
    # print len(train_x)
    # print len(train_y)
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    label_rate = check_balance_data(train_y, train_y)
    balance = False
    if max(label_rate.keys()) > balance_rate and balanc_data:
        print('The largest category accounted for more than %s Start balancing data' % (str(balance_rate * 100) + '%'))
        train_x, train_y = balance_data(train_x, train_y)
        balance = True
        np.save(os.path.join(output, 'balance_train_x.npy'), train_x)
        np.save(os.path.join(output, 'balance_trian_y.npy'), train_y)
        np.save(os.path.join(output, 'balance_test_x.npy'), test_x)
        np.save(os.path.join(output, 'balance_test_y.npy'), test_y)
    else:
        np.save(os.path.join(output, 'train_x.npy'), train_x)
        np.save(os.path.join(output, 'trian_y.npy'), train_y)
        np.save(os.path.join(output, 'test_x.npy'), test_x)
        np.save(os.path.join(output, 'test_y.npy'), test_y)

    print('Start sliding window for data')
    train_x, train_y = opp_sliding_window(train_x, train_y, ws, ss)
    test_x, test_y = opp_sliding_window(test_x, test_y, ws, ss)

    # Dataset is segmented into train and test
    # nb_training_samples = 557963
    # The first 18 OPPORTUNITY data files define the traning dataset, comprising 557963 samples
    # X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    # X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]

    # print "Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape)

    # obj = [(X_train, y_train), (X_test, y_test)]

    # train_obj = (train_x, train_y)
    # test_obj = (test_x, test_y)
    # with file(os.path.join(target_filename, str(ws) + '_' + str(ss) + '_oppor_train_data.npy'), 'wb') as f:
    #     cp.dump(train_obj, f, protocol=cp.HIGHEST_PROTOCOL)
    # with file(os.path.join(target_filename, str(ws) + '_' + str(ss) + '_oppor_test_data.npy'), 'wb') as f:
    #     cp.dump(test_obj, f, protocol=cp.HIGHEST_PROTOCOL)

    train_y = np.vstack([stats.mode(label)[0] for label in train_y])
    test_y = np.vstack([stats.mode(label)[0] for label in test_y])

    print('The train_x shape is:', train_x.shape)
    print('The train_y shape is:', train_y.shape)
    print('The test_x shape is:', test_x.shape)
    print('The test_y shape is:', test_y.shape)

    file_dic = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}
    if balance:
        for name, data in file_dic.items():
            print(name, data.shape[0])
            out_path = os.path.join(target_filename,
                                    'balance_' + str(ws) + '_' + str(ss) + '_opprtuntiy_' + name + '.npy')
            np.save(out_path, data)
        train_x = train_x.tolist()
        train_y = train_y.tolist()
        test_x = test_x.tolist()
        test_y = test_y.tolist()

        train_x = pd.DataFrame(train_x)
        test_x = pd.DataFrame(test_x)

        train_x['label'] = train_y
        test_x['label'] = test_y

        train_x.to_csv(os.path.join(target_filename, 'balance_' + str(ws) + '_' + str(ss) + '_opprtuntiy_train.csv'))
        test_x.to_csv(os.path.join(target_filename, 'balance_' + str(ws) + '_' + str(ss) + '_opprtuntiy_test.csv'))

    else:
        for name, data in file_dic.items():
            print(name, data.shape[0])
            out_path = os.path.join(target_filename,
                                    str(ws) + '_' + str(ss) + '_opprtuntiy_' + name + '.npy')
            np.save(out_path, data)
        train_x = train_x.tolist()
        train_y = train_y.tolist()
        test_x = test_x.tolist()
        test_y = test_y.tolist()

        train_x = pd.DataFrame(train_x)
        test_x = pd.DataFrame(test_x)

        train_x['label'] = train_y
        test_x['label'] = test_y

        train_x.to_csv(os.path.join(target_filename, str(ws) + '_' + str(ss) + '_opprtuntiy_train.csv'))
        test_x.to_csv(os.path.join(target_filename, str(ws) + '_' + str(ss) + '_opprtuntiy_test.csv'))

    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)


if __name__ == '__main__':
    if is_linux:
        data_dir = r'/home/lyk/OpportunityUCIDataset/dataset'
        output = r'/home/lyk/prepared_data/opprtunity'
    else:
        data_dir = r'G:/data/OpportunityUCIDataset/dataset'
        output = r'G:/data/prepared_data/opprtunity'
    balance_rate = 0.4
    l = 'gestures'
    ws = 24
    ss = 24
    generate_data(data_dir, output, l, ws, ss)
