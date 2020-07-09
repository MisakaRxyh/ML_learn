import sys
import pandas as pd
import numpy as np
import math
import csv

data = pd.read_csv('./ml2020spring-hw1/train.csv', encoding='big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            # x 输入值，将每9个小时的18个特征的数据合成了一行，每次小时数+1再取一行，[0,9][1,10][2,11]...
            # 每个月20天有480个小时，每个月选取480 - 9 = 471组数据，一年共有471 * 12组数据
            # 一组数据里有9 * 18个数值
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                                     -1)
            # y 输出值，将每次取的第10个小时的PM2.5值取出，[10][11][12]...
            # 每个月选取471组测试数据，一年共有471*12组数据
            # 一组数据里有 1 个数据，即PM2.5的值
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

# Normalize 标准化 mean 期望 std 标准差
mean_x = np.mean(x, axis=0)  # 求471 * 12组数据的期望
std_x = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
# 选取 x,y 的前80%作为训练集， 80%之后作为验证集， 测试集并未导入
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))


# Training
# dim + 1是因为有常数项的存在
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)

learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.00000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12) # RMSE 均方根误差
    if(t % 100 == 0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) -y)
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
# print(w)


#Testing
test_data = pd.read_csv('./ml2020spring-hw1/test.csv', header=None, encoding='big5')
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 *(i + 1), :].reshape(1,-1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)


# Prediction
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

print(ans_y)


# Save Prediction
with open('submit.csv', mode='w', newline='') as submit_flie:
    csv_writer = csv.writer(submit_flie)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)