import os
import matplotlib.pyplot as plt
import numpy as np

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

def plot(container, classification, num):
    plt.plot(container)
#    plt.ylim(0,max(credibilityDis[i]) + 0.002) 
#    plt.plot(credibilityDis[i])
    plt.savefig('vision/%s/%s' % (classification, num))  
    plt.cla()


def normalization(data):
    data_max = max(data)
    data_min = min(data)
    data_range = data_max - data_min
    data = list(map(lambda x: (x - data_min)/data_range, data))    
    return data

def dimstandard(data):
    data = list(map(convert, data))  
#    print(len(data))
#    data_new = []
#    for i in range(len(all_gruop)):
#        for item in all_gruop[i]:
#            data_new.append(item)
    data_new = list(map(normalization, data))
#    data_final = []
#    count = 0
#    for i in range(len(data_new) + 1):
#        if i > 0 and i % 4000 == 0:
#            data_final.append(data_new[count: i])
#            count = i
    data_final = list(map(lambda x: np.array(x), data_new))
    
    return data_final

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
    aimFile = "VIBMON_K1702B_1H.txt"
    origin = readfile(aimFile)
#    print(len(origin))
    data = dimstandard(origin)

    for num in range(len(data)):
        plot(data[num], 'original', num)
    data = list(map(np.fft.fft, data))
#    print(data[0])
    for num in range(len(data)):
        plot(data[num], 'fft', num)