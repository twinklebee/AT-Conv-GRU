import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.recurrent import GRU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import numpy as np
from data_preparation import *
from utils import *
from keras.layers import Bidirectional
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import LSTM

from keras.models import *
from keras.layers.convolutional import Conv1D
from attention import AttentionLayer
from attention_with_context import AttentionWithContext
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint

import time

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
print('loading data...')
# data1 = load_csv(r'data-freeway/10105110', 8, "freeway")
# data2 = load_csv(r'data-freeway/10105310', 8, "freeway")
# data3 = load_csv(r'data-freeway/10105510', 8, "freeway")
# data4 = load_csv(r'data-freeway/10108210', 8, "freeway")
# data5 = load_csv(r'data-freeway/10106510', 8, "freeway")
# data6 = load_csv(r'data-freeway/1095110', 8, "freeway")
# data7 = load_csv(r'data-freeway/1095510', 8, "freeway")
#
# data1_h = load_csv(r'data-freeway/10105110', 12, "freeway")
# data2_h = load_csv(r'data-freeway/10105310', 12, "freeway")
# data3_h = load_csv(r'data-freeway/10105510', 12, "freeway")
# data4_h = load_csv(r'data-freeway/10108210', 12, "freeway")
# data5_h = load_csv(r'data-freeway/10106510', 12, "freeway")
# data6_h = load_csv(r'data-freeway/1095110', 12, "freeway")
# data7_h = load_csv(r'data-freeway/1095510', 12, "freeway")
#
# data1_lane = load_csv(r'data-freeway/10105110', 10, "freeway")
# data2_lane = load_csv(r'data-freeway/10105310', 10, "freeway")
# data3_lane = load_csv(r'data-freeway/10105510', 10, "freeway")
# data4_lane = load_csv(r'data-freeway/10108210', 10, "freeway")
# data5_lane = load_csv(r'data-freeway/10106510', 10, "freeway")
# data6_lane = load_csv(r'data-freeway/1095110', 10, "freeway")
# data7_lane = load_csv(r'data-freeway/1095510', 10, "freeway")

data1 = load_csv(r'data-urban/401190', 5, "urban")
data2 = load_csv(r'data-urban/401144', 7, "urban")
data3 = load_csv(r'data-urban/401413', 11, "urban")
data4 = load_csv(r'data-urban/401911', 8, "urban")
data5 = load_csv(r'data-urban/401610', 10, "urban")
data6 = load_csv(r'data-urban/401273', 8, "urban")
data7 = load_csv(r'data-urban/401137', 8, "urban")

data1_lane = load_csv(r'data-urban/401190', 7, "urban")
data2_lane = load_csv(r'data-urban/401144', 9, "urban")
data3_lane = load_csv(r'data-urban/401413', 13, "urban")
data4_lane = load_csv(r'data-urban/401911', 10, "urban")
data5_lane = load_csv(r'data-urban/401610', 12, "urban")
data6_lane = load_csv(r'data-urban/401273', 10, "urban")
data7_lane = load_csv(r'data-urban/401137', 10, "urban")

data1_h = load_csv(r'data-urban/401190', 9, "urban")
data2_h = load_csv(r'data-urban/401144', 11, "urban")
data3_h = load_csv(r'data-urban/401413', 15, "urban")
data4_h = load_csv(r'data-urban/401911', 12, "urban")
data5_h = load_csv(r'data-urban/401610', 14, "urban")
data6_h = load_csv(r'data-urban/401273', 12, "urban")
data7_h = load_csv(r'data-urban/401137', 12, "urban")

epoch = 50
day = 288
week = 2016
seq_len = 15
# 1=5min, 3=15min, 6=30min, 12=60min
pre_len = 12
# data 1-7
pre_sens_num = 2

# train,test
train_data, train_w, train_d, label, test_data, test_w, test_d, test_l, test_med, test_min \
    = generate_data(data1, data2, data3, data4, data5, data6, data7, seq_len, pre_len, pre_sens_num)

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
train_w = np.reshape(train_w, (train_w.shape[0], train_w.shape[1], 1))
train_d = np.reshape(train_d, (train_d.shape[0], train_d.shape[1], 1))

test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
test_d = np.reshape(test_d, (test_d.shape[0], test_d.shape[1], 1))
test_w = np.reshape(test_w, (test_w.shape[0], test_w.shape[1], 1))

# LCS
train_data1, train_w1, train_d1, label1, test_data1, test_w1, test_d1, test_l1, test_med1, test_min1 \
    = generate_data(data1_lane, data2_lane, data3_lane, data4_lane, data5_lane, data6_lane, data7_lane, seq_len,
                    pre_len, pre_sens_num)

train_data1 = np.reshape(train_data1, (train_data1.shape[0], train_data1.shape[1], train_data1.shape[2], 1))
train_w1 = np.reshape(train_w1, (train_w1.shape[0], train_w1.shape[1], 1))
train_d1 = np.reshape(train_d1, (train_d1.shape[0], train_d1.shape[1], 1))

test_data1 = np.reshape(test_data1, (test_data1.shape[0], test_data1.shape[1], test_data1.shape[2], 1))
test_d1 = np.reshape(test_d1, (test_d1.shape[0], test_d1.shape[1], 1))
test_w1 = np.reshape(test_w1, (test_w1.shape[0], test_w1.shape[1], 1))

# holiday
train_data2, train_w2, train_d2, label2, test_data2, test_w2, test_d2, test_l2, test_med2, test_min2 \
    = generate_data(data1_h, data2_h, data3_h, data4_h, data5_h, data6_h, data7_h, seq_len,
                    pre_len, pre_sens_num)

train_data2 = np.reshape(train_data2, (train_data2.shape[0], train_data2.shape[1], train_data2.shape[2], 1))
train_w2 = np.reshape(train_w2, (train_w2.shape[0], train_w2.shape[1], 1))
train_d2 = np.reshape(train_d2, (train_d2.shape[0], train_d2.shape[1], 1))

test_data2 = np.reshape(test_data2, (test_data2.shape[0], test_data2.shape[1], test_data2.shape[2], 1))
test_d2 = np.reshape(test_d2, (test_d2.shape[0], test_d2.shape[1], 1))
test_w2 = np.reshape(test_w2, (test_w2.shape[0], test_w2.shape[1], 1))

# CNN-Lane
# main_input1 = Input((15, 7, 1), name='main_input1')
# con11 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(main_input1)
# con22 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(con11)
#
# con_fl1 = TimeDistributed(Flatten())(con22)
#
# con_out_lane = Dense(15)(con_fl1)
# con_out_lane1 = Dense(5)(con_out_lane)
# con_fl2 = Flatten()(con_out_lane1)

# con_fl2 = Flatten()(con_out_lane)

# CNN-holiday
main_input2 = Input((15, 7, 1), name='main_input2')
con33 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(main_input2)
con44 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(con33)

con_fl3 = TimeDistributed(Flatten())(con44)

con_out_h = Dense(15)(con_fl3)
con_out_h1 = Dense(5)(con_out_h)
con_fl4 = Flatten()(con_out_h1)

# con_out_lane = Dense(15)(con_fl1)
# lstm_out1_lane = LSTM(15, return_sequences=True)(con_out_lane)
# lstm_attention_lane = AttentionWithContext()(lstm_out1_lane)
# lstm_out2_lane = LSTM(15, return_sequences=False)(lstm_attention_lane)
# lstm_out3_lane = AttentionLayer()([lstm_out2_lane, con_out_lane])


# conv-gru
main_input = Input((15, 7, 1), name='main_input')
con1 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(main_input)
con2 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(con1)
# con3 = TimeDistributed(AveragePooling1D(pool_size=2))(con2)
con_fl = TimeDistributed(Flatten())(con2)
con_out = Dense(15)(con_fl)

gru_out1 = GRU(15, return_sequences=True, dropout=0.01, recurrent_dropout=0.01)(con_out)
gru_attention = AttentionWithContext()(gru_out1)
gru_out2 = GRU(15, return_sequences=False, dropout=0.01, recurrent_dropout=0.01)(gru_attention)
gru_out3 = AttentionLayer()([gru_out2, con_out])

# BiGRU
auxiliary_input_w = Input((15, 1), name='auxiliary_input_w')
gru_outw1 = Bidirectional(GRU(15, return_sequences=True, dropout=0.01, recurrent_dropout=0.01))(auxiliary_input_w)
gru_outw2 = Bidirectional(GRU(15, return_sequences=False, dropout=0.01, recurrent_dropout=0.01))(gru_outw1)

auxiliary_input_d = Input((15, 1), name='auxiliary_input_d')
gru_outd1 = Bidirectional(GRU(15, return_sequences=True, dropout=0.01, recurrent_dropout=0.01))(auxiliary_input_d)
gru_outd2 = Bidirectional(GRU(15, return_sequences=False, dropout=0.01, recurrent_dropout=0.01))(gru_outd1)

x = keras.layers.concatenate([ con_fl4, gru_out3, gru_outw2, gru_outd2])
x = Dense(20, activation='relu')(x)
x = Dense(10, activation='relu')(x)
main_output = Dense(1, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(0.1, 0.1), name='main_output')(x)
model = Model(inputs=[main_input2, main_input, auxiliary_input_w, auxiliary_input_d],
              outputs=main_output)
model.summary()
model.compile(optimizer='adam', loss=my_loss)

# train_save model
filepath = "model/model_{epoch:04d}-{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min', period=5)
callbacks_list = [checkpoint]

time_start = time.time()

print('Train...')
model.fit([ train_data2, train_data, train_w, train_d], label,
          batch_size=128, epochs=epoch, validation_split=0.3, verbose=2,
          class_weight='auto', callbacks=callbacks_list)
model_json = model.to_json()
with open("model/conv_lstm.json", "w") as json_file:
    json_file.write(model_json)
print("Save model to disk")

# load model
# with CustomObjectScope({'AttentionLayer': AttentionLayer}):
# 	json_file = open('model/conv_lstm.json', 'r')
# 	loaded_model_json = json_file.read()
# 	json_file.close()
# 	cnn_lstm_model = model_from_json(loaded_model_json)
# 	cnn_lstm_model.load_weights("model/model_0200-0.0337.h5", 'r')

# Predict the last model
predicted = predict_point_by_point(model, [ test_data2, test_data, test_w, test_d])
print(predicted.shape)
p_real = []
l_real = []
p_real1 = []
l_real1 = []
row = 2016
for i in range(row):
    p_real.append(predicted[i] * test_med + test_min)
    l_real.append(test_l[i] * test_med + test_min)

for i in range(288):
    p_real1.append((p_real[i] + p_real[i + 288] + p_real[i + 288 * 2] + p_real[i + 288 * 3] + p_real[i + 288 * 4] +
                    p_real[i + 288 * 5] + p_real[i + 288 * 6]) / 7)
    l_real1.append((
                           l_real[i] + l_real[i + 288] + l_real[i + 288 * 2] + l_real[i + 288 * 3] + l_real[
                       i + 288 * 4] + l_real[
                               i + 288 * 5] + l_real[i + 288 * 6]) / 7)
p_real = np.array(p_real)
l_real = np.array(l_real)

p_real2 = p_real1[0:144]
l_real2 = l_real1[0:144]

p_real2 = np.array(p_real2)
l_real2 = np.array(l_real2)
# p_real1 = np.array(p_real1)
# l_real1 = np.array(l_real1)
# draw figure of real and predict
data_p_real = pd.DataFrame(p_real2)
data_l_real = pd.DataFrame(l_real2)

data_p_real.to_csv('p_real.csv', index=None)
data_l_real.to_csv('l_real.csv', index=None)

plt.figure(figsize=(18, 10))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel("Time(h)", size=28)
plt.ylabel("Traffic Flow(veh)", size=28)
x = []
for i in range(144):  # number of x-axis
    x.append(i / 12)
plt.plot(x, l_real2, c="r", label="Real Traffic Data", linewidth=2)
plt.plot(x, p_real2, c="g", label="Predicted Traffic Data", linewidth=2)
# for i in range(1728, 2016, 1):  # number of x-axis
#     x.append(i)
#     plt.plot(x, l_real[1728:2016:1], c="r", label="Real Traffic Data", linewidth=2)
#     plt.plot(x, p_real[1728:2016:1], c="g", label="Predicted Traffic Data", linewidth=2)
plt.legend(loc='upper left', fontsize='xx-large')
from datetime import datetime

now = datetime.now()
now = now.strftime("%m%d%H%M%S")
# 指定图片保存路径
figure_save_path = "cmp-fig"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
plt.savefig(os.path.join(figure_save_path, 'Fig' + str(now) + '.png'))  # 第一个是指存储路径，第二个是图片名字
# plt.show()

time_end=time.time()
run_time=time_end-time_start
print(run_time)

print("MAE:", MAE(p_real, l_real))
print("MAPE:", MAPE(p_real, l_real))
print("RMSE:", RMSE(p_real, l_real))
