import os

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import numpy as np
from data_preparation import *
import time
from utils import *
from keras.layers import Bidirectional
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.models import *
from keras.layers.convolutional import Conv1D
from attention import AttentionLayer
from attention_with_context import AttentionWithContext
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import GRU

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
print('loading data...')
# data1 = load_csv(r'data-freeway/10105110', 8, "freeway")
# data2 = load_csv(r'data-freeway/10105310', 8, "freeway")
# data3 = load_csv(r'data-freeway/10105510', 8, "freeway")
# data4 = load_csv(r'data-freeway/10108210', 8, "freeway")
# data5 = load_csv(r'data-freeway/10106510', 8, "freeway")
# data6 = load_csv(r'data-freeway/1095110', 8, "freeway")
# data7 = load_csv(r'data-freeway/1095510', 8, "freeway")

his=3
#1,3,6,12



data2 = load_csv(r'data-urban/401144', 7, "urban")
data2_lane = load_csv(r'data-urban/401144', 9, "urban")
data2_h = load_csv(r'data-urban/401144', 11, "urban")
data=np.stack((data2,data2_lane,data2_h),axis=1)
max = np.max(data)
seq_len = 15
pre_sens_num = 1
max = np.max(data)
min = np.min(data)
med = max - min
data = np.array(data, dtype=float)
data_nor = (data - min) / med
#print("data_nor", data_nor)
sequence_length = seq_len + his
result = []
for index in range(len(data_nor) - sequence_length):
    result.append(data_nor[index: index + sequence_length])
result = np.stack(result, axis=0)
#print("result", result)
train = result[:]
x_train = train[:, :seq_len]
#print("x_train", x_train)
y_train = train[:, -1, pre_sens_num - 1]
#print("y_train", y_train)
x_data = []
label = []
for i in range(len(train)):
    if i >= 8640:
        x_data.append(x_train[i])
        label.append(y_train[i])
x_data = np.array(x_data)
label = np.array(label)
row = 8640
row_test = 2016
train_x_data = x_data[:-row]
test_data = x_data[-row_test:]
train_l = label[:-row]
test_l = label[-row_test:]
train_num=int(0.85*len(train_x_data))
train_x = train_x_data[:train_num]
train_y = train_l[:train_num]
val_x,val_y = train_x[train_num:],train_l[train_num:]

json_file = open('model/conv_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn_lstm_model = model_from_json(loaded_model_json)
cnn_lstm_model.load_weights('model/model_0035-0.0067.h5', 'r')

# start =time.clock()
predicted = predict_point_by_point(cnn_lstm_model, test_data)

p_real = []
l_real = []
row = 2016
for i in range(row):
    p_real.append(predicted[i] * med + min)
    l_real.append(test_l[i] * med + min)
p_real = np.array(p_real)
l_real = np.array(l_real)

# draw figure of real and predict
data_p_real = pd.DataFrame(p_real)
data_l_real = pd.DataFrame(l_real)

data_p_real.to_csv('p_real.csv', index=None)
data_l_real.to_csv('l_real.csv', index=None)

print("MAE:", MAE(p_real, l_real))
print("MAPE:", MAPE(p_real, l_real))
print("RMSE:", RMSE(p_real, l_real))

# end = time.clock()
