import os
import matplotlib.pyplot as plt
import numpy as np
import random

def readfile(aimFile):
    try:
        with open(aimFile, 'r+') as f:
            b = []
            while True:
                a = f.readline()
                if not a:
                    break
                b.append(a)
    except Exception as e:
        raise e
    return b

def convert(b):
    b = b.replace(' ', ',')
    count = -1
    container = []
    for i in range(len(b)):
        if b[i] == ',':
            new_item = int(b[count + 1:i])
            count = i
            container.append(new_item)
    return container[: 4000]

def normalization(data):
    data_max = max(data)
    data_min = min(data)
    data_range = data_max - data_min
    data = list(map(lambda x: ((x - data_min)/data_range), data))    
    return data

def dimstandard(data):
    all_gruop = list(map(convert, data))  
    data_new = list(map(normalization, all_gruop))
    data_final = list(map(lambda x: np.array(x), data_new))
    
    return data_final


# ----------标准的批量训练数据读取方式---------------#
# ----------yield生成器，每次只读取所需批次的数据----#
# ----------由于数据量较少一次性读取所有数据占用的空间也不大，故暂未采用，后续完善再补上------#
def next_batch(data, batch_size, num_epochs, shuffle=True):
    data = np.array(sourceData)  # 将sourceData转换为array存储
    data_size = len(sourceData)
    num_batches_per_epoch = int(len(sourceData) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = sourceData[shuffle_indices]
        else:
            shuffled_data = sourceData

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    # exit()
    aimFile = "VIBMON_K1702B_1H.txt"
    origin = readfile(aimFile)
    data = dimstandard(origin)
    # print(len(data))