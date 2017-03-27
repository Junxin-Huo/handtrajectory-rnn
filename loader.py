import numpy as np
import os
import random

FRAME_COUNT = 10
VAR_LABEL = 3
CAMERA_RESOLUTION = 480

def loadDataLabel(dir_name, shuffle=False, various=False):
    assert os.path.isdir(dir_name), "dir_name is not dir"
    dir = os.listdir(dir_name)
    dir.sort()
    len_dir = len(dir)
    datas = []
    labels = []
    for i in range(len_dir):
        if os.path.isdir(dir_name + '/' + dir[i]):
            continue
        flag = False
        if dir[i] == 'hjx.txt' or dir[i] == 'yjl.txt' or dir[i] == 'mml.txt':
            flag = True
        f = open(dir_name + '/' + dir[i], "r")
        lines = f.readlines()
        for line in lines:
            words = line.split(' ')
            if flag:
                len_frame = (len(words) - 3) / 3
                if len_frame < FRAME_COUNT:
                    continue

                label = int(words[1])
                split_words = np.asarray(words[2:len(words) - 1], dtype=np.float)
                split_words = np.reshape(split_words, [-1, 3])
                temp0, temp1, temp2 = np.split(split_words.T, 3)

                x = np.zeros((len_frame + 1, 2), dtype=np.float)
                for j in range(len_frame):
                    x[j + 1, 0] = x[j, 0] + temp1[0, j]
                    x[j + 1, 1] = x[j, 1] + temp2[0, j]

                temp = np.linspace(0, len_frame, FRAME_COUNT + 1)
                for k in range(len(temp)):
                    temp[k] = round(temp[k])
                data = np.zeros([FRAME_COUNT, 2], dtype=np.float)
                y = np.zeros(FRAME_COUNT, dtype=np.float)
                for k in range(FRAME_COUNT):
                    data[k, 0] = x[int(temp[k + 1]), 0] - x[int(temp[k]), 0]
                    data[k, 1] = x[int(temp[k + 1]), 1] - x[int(temp[k]), 1]
                    y[k] = temp0[0, int(temp[k + 1]) - 1]

            else:
                len_frame = (len(words) - 3) / 3
                if len_frame < FRAME_COUNT:
                    continue

                label = int(words[1])
                split_words = np.asarray(words[2:len(words) - 1], dtype=np.float)
                split_words = np.reshape(split_words, [-1, 3])
                temp0, temp1, temp2 = np.split(split_words.T, 3)

                x = np.zeros((len_frame, 2), dtype=np.float)
                for j in range(len_frame):
                    x[j, 0] = temp1[0, j]
                    x[j, 1] = temp2[0, j]

                temp = np.linspace(0, len_frame - 1, FRAME_COUNT + 1)
                for k in range(len(temp)):
                    temp[k] = round(temp[k])
                data = np.zeros([FRAME_COUNT, 2], dtype=np.float)
                y = np.zeros(FRAME_COUNT, dtype=np.float)
                for k in range(FRAME_COUNT):
                    data[k, 0] = x[int(temp[k + 1]), 0] - x[int(temp[k]), 0]
                    data[k, 1] = x[int(temp[k + 1]), 1] - x[int(temp[k]), 1]
                    y[k] = temp0[0, int(temp[k + 1])]

                data = data / CAMERA_RESOLUTION


            datas.append(data)
            labels.append(dataRelabel(y, label=label))
            if various:
                if label == 2:
                    var_label = dataRelabel(y, label=2)
                elif label == 3:
                    var_label = dataRelabel(y, label=3)
                else:
                    var_label = dataRelabel(y, label=label)
                hor_data = horizontal(data)
                ver_data = vertical(data)
                hv_data = vertical(hor_data)

                datas.append(hor_data)
                datas.append(ver_data)
                datas.append(hv_data)
                labels.append(var_label)
                labels.append(var_label)
                labels.append(dataRelabel(y, label=label))
        f.close()

    if shuffle:
        print("Shuffling...")
        index = range(len(labels))
        random.shuffle(index)
        xx = []
        yy = []
        for i in range(len(labels)):
            xx.append(datas[index[i]])
            yy.append(labels[index[i]])
        datas = xx
        labels = yy
    return np.asarray(datas, np.float32), np.asarray(labels, np.float32)


def horizontal(data):
    length = len(data)
    result = np.zeros_like(data)
    for i in range(length):
        result[i, 0] = data[i, 0] * -1
        result[i, 1] = data[i, 1]
    return result


def vertical(data):
    length = len(data)
    result = np.zeros_like(data)
    for i in range(length):
        result[i, 0] = data[i, 0]
        result[i, 1] = data[i, 1] * -1
    return result


def switchXY(data):
    length = len(data)
    result = np.zeros_like(data)
    for i in range(length):
        result[i, 0] = data[i, 1]
        result[i, 1] = data[i, 0]
    return result


def dataRelabel(data, label):
    data_temp = data.copy()
    for i in range(len(data_temp)):
        if data_temp[i] == 0:
            data_temp[i] = VAR_LABEL
        else:
            data_temp[i] = label
    return data_temp


def loadDataLabelRealtime(dir_name, shuffle=False, various=False):
    assert os.path.isdir(dir_name), "dir_name is not dir"
    dir = os.listdir(dir_name)
    dir.sort()
    len_dir = len(dir)
    datas = []
    labels = []
    for i in range(len_dir):
        if os.path.isdir(dir_name + '/' + dir[i]):
            continue
        flag = False
        if dir[i] == 'hjx.txt' or dir[i] == 'yjl.txt' or dir[i] == 'mml.txt':
            flag = True
        f = open(dir_name + '/' + dir[i], "r")
        lines = f.readlines()
        for line in lines:
            words = line.split(' ')
            if flag:
                len_frame = (len(words) - 3) / 3
                # if len_frame < FRAME_COUNT:
                #     continue

                label = int(words[1])
                split_words = np.asarray(words[2:len(words) - 1], dtype=np.float)
                split_words = np.reshape(split_words, [-1, 3])
                temp0, temp1, temp2 = np.split(split_words.T, 3)

                x = np.zeros((len_frame + 1, 2), dtype=np.float)
                for j in range(len_frame):
                    x[j + 1, 0] = x[j, 0] + temp1[0, j]
                    x[j + 1, 1] = x[j, 1] + temp2[0, j]

                data = np.zeros([len_frame, 2], dtype=np.float)
                y = np.zeros(len_frame, dtype=np.float)
                for k in range(len_frame):
                    data[k, 0] = x[k + 1, 0] - x[k, 0]
                    data[k, 1] = x[k + 1, 1] - x[k, 1]
                    y[k] = temp0[0, k]

            else:
                len_frame = (len(words) - 3) / 3
                # if len_frame < FRAME_COUNT:
                #     continue

                label = int(words[1])
                split_words = np.asarray(words[2:len(words) - 1], dtype=np.float)
                split_words = np.reshape(split_words, [-1, 3])
                temp0, temp1, temp2 = np.split(split_words.T, 3)

                x = np.zeros((len_frame, 2), dtype=np.float)
                for j in range(len_frame):
                    x[j, 0] = temp1[0, j]
                    x[j, 1] = temp2[0, j]

                data = np.zeros([len_frame - 1, 2], dtype=np.float)
                y = np.zeros(len_frame - 1, dtype=np.float)
                for k in range(len_frame - 1):
                    data[k, 0] = x[k + 1, 0] - x[k, 0]
                    data[k, 1] = x[k + 1, 1] - x[k, 1]
                    y[k] = temp0[0, k]

                data = data / CAMERA_RESOLUTION


            datas.append(data)
            labels.append(dataRelabel(y, label=label))

            if various:
                if label == 2:
                    var_label = dataRelabel(y, label=2)
                elif label == 3:
                    var_label = dataRelabel(y, label=3)
                else:
                    var_label = dataRelabel(y, label=label)
                hor_data = horizontal(data)
                ver_data = vertical(data)
                hv_data = vertical(hor_data)

                datas.append(hor_data)
                datas.append(ver_data)
                datas.append(hv_data)
                labels.append(var_label)
                labels.append(var_label)
                labels.append(dataRelabel(y, label=label))

        f.close()

    if shuffle:
        print("Shuffling...")
        index = range(len(labels))
        random.shuffle(index)
        xx = []
        yy = []
        for i in range(len(labels)):
            xx.append(datas[index[i]])
            yy.append(labels[index[i]])
        datas = xx
        labels = yy
    return datas, labels


def loadDataLabelSequence(dir_name, batch_size):
    dir_name = dir_name + '/sequence'
    assert os.path.isdir(dir_name), "dir_name is not dir"
    dir = os.listdir(dir_name)
    dir.sort()
    len_dir = len(dir)
    datas = []
    labels = []
    for i in range(len_dir):
        if os.path.isdir(dir_name + '/' + dir[i]):
            continue
        f = open(dir_name + '/' + dir[i], "r")
        lines = f.readlines()
        for line in lines:
            words = line.split(' ')
            len_frame = (len(words) - 2) / 4

            split_words = np.asarray(words[1:len(words) - 1], dtype=np.float)
            split_words = np.reshape(split_words, [-1, 4])
            temp0, temp1, temp2, temp3 = np.split(split_words.T, 4)

            point = np.zeros((len_frame, 2), dtype=np.float)
            for j in range(len_frame):
                point[j, 0] = temp2[0, j]
                point[j, 1] = temp3[0, j]

            for k in range(len_frame - 1):
                x = np.zeros([2], dtype=np.float)
                x[0] = (point[k + 1, 0] - point[k, 0]) / CAMERA_RESOLUTION
                x[1] = (point[k + 1, 1] - point[k, 1]) / CAMERA_RESOLUTION
                if temp1[0, k + 1] == 255:
                    y = VAR_LABEL
                else:
                    y = temp1[0, k + 1]

                datas.append(x)
                labels.append(y)

        f.close()

    data_len = len(labels)
    batch_len = data_len // batch_size
    epoch_size = batch_len // FRAME_COUNT
    datas = np.asarray(datas, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    _datas = np.reshape(datas[0: FRAME_COUNT * epoch_size * batch_size, ...], [batch_size, epoch_size, FRAME_COUNT, 2])
    _labels = np.reshape(labels[0: FRAME_COUNT * epoch_size * batch_size], [batch_size, epoch_size, FRAME_COUNT])


    return _datas, _labels