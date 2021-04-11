import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import metrics




 # データの読み込み
df = pd.read_csv('preprocess/saitama.csv')

name = 'predict'

# 特定のカラムの抽出
df_new = df.loc[:, [name]]

# 折れ線グラフの描画
df_new.plot(kind='line')

# 描画の実行
plt.show()


df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")

# データの並び替え
df.sort_values(by='Date', ascending=True, inplace=True)

# インデックスの更新
df.set_index(keys='Date', inplace=True)

# 説明変数をX_dataに格納
X_data = df.drop(columns=[name],inplace=False)
X_data = np.array(X_data)
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)


# 目的変数をy_dataに格納
y_data = df[name]
y_data = np.array(y_data)
y_data = y_data.reshape(-1, 1)
scaler1 = StandardScaler()
y_data = scaler1.fit_transform(y_data)

# 学習データおよび検証データと、評価データに80:20の割合で2分割する
X_trainval, X_test, y_trainval, y_test = train_test_split(X_data, y_data, test_size=0.10, shuffle=False)

# 学習データと検証データに75:25の割合で2分割する
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, shuffle=False)


def get_standardized_t(X, num_date):
   
    X_t_list = []
    for i in range(len(X) - num_date + 1):
        X_t = X[i:i+num_date]
       
        X_t_list.append(X_t)
    return np.array(X_t_list)

def get_standardized_y_t(y, num_date):
    y_t = y[num_date-1 : ]
    return y_t

num_date = 7

# 学習用、検証用、評価用データの加工
X_train_t =  get_standardized_t(X=X_train, num_date=num_date)
X_val_t = get_standardized_t(X=X_val, num_date=num_date)
X_test_t = get_standardized_t(X=X_test, num_date=num_date)

# 目的変数の変形
y_train_t = get_standardized_y_t(y=y_train, num_date=num_date)
y_val_t = get_standardized_y_t(y=y_val, num_date=num_date)
y_test_t = get_standardized_y_t(y=y_test, num_date=num_date)

num_l1 = 300
num_l2 = 100
num_output = 1

model = Sequential()

model.add(LSTM(units=num_l1,
                activation='tanh',
                input_shape=(X_train_t.shape[1], X_train_t.shape[2])))

model.add(Dense(num_l2))

model.add(Dense(num_output))

model.summary()

model.compile(loss='MAPE', optimizer='adam', metrics=[metrics.mae])

result = model.fit(x=X_train_t, y=y_train_t, epochs=100, batch_size=64, validation_data=(X_val_t, y_val_t))

model.save('./model/lstm_model.h5')
model.save_weights('./weights/lstm_weights.h5')


# 折れ線グラフによる学習データの損失の描画
plt.plot(result.history['mean_absolute_error'])

# 折れ線グラフによる検証データの損失の描画
plt.plot(result.history['val_mean_absolute_error'])

# 凡例の指定
plt.legend(['Train', 'Val'])

# グラフの軸タイトルの指定
plt.xlabel('Epoch')
plt.ylabel('MAE')

# 描画の実行
plt.show()


predict = model.predict(X_test_t)

y_train = scaler1.inverse_transform(y_train_t)
y_train = pd.DataFrame(y_train)

y_test = scaler1.inverse_transform(y_test_t)
y_test = pd.DataFrame(y_test)

predict = scaler1.inverse_transform(predict)
print(y_test)
print(predict)
Predict = pd.DataFrame(predict)

plt.figure(figsize=(15,10))
plt.plot(y_test, label = 'Test')
plt.plot(Predict, label = 'Prediction')
plt.legend(loc='best')
plt.show()