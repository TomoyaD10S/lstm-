import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def preprocess():

    # データの読み込み
    df = pd.read_csv('preprocess/hiroshima.csv')

    name = 'predict'

    def transform():
        df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")

        df.sort_values(by='Date', ascending=True, inplace=True)
        df.set_index(keys='Date', inplace=True)

        X_data = df.drop(columns=[name],inplace=False)
        X_data = np.array(X_data)
        scaler = StandardScaler()
        X_data = scaler.fit_transform(X_data)

        y_data = df[name]
        y_data = np.array(y_data)
        y_data = y_data.reshape(-1, 1)
        scaler1 = StandardScaler()
        y_data = scaler1.fit_transform(y_data)
        
        X_trainval, X_test, y_trainval, y_test = train_test_split(X_data, y_data, test_size=0.20, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, shuffle=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    X_train, X_val, X_test, y_train, y_val, y_test = transform()


    def transform_X(X, num_date):
        X_t_list = []
        for i in range(len(X) - num_date + 1):
            X_t = X[i:i+num_date]
            X_t_list.append(X_t)
        return np.array(X_t_list)

    def transform_y(y, num_date):
        y_t = y[num_date-1 : ]
        return y_t

    num_date = 7

    # 学習用、検証用、評価用データの加工
    X_train_t =  transform_X(X=X_train, num_date=num_date)
    X_val_t = transform_X(X=X_val, num_date=num_date)
    X_test_t = transform_X(X=X_test, num_date=num_date)

    # 目的変数の変形
    y_train_t = transform_y(y=y_train, num_date=num_date)
    y_val_t = transform_y(y=y_val, num_date=num_date)
    y_test_t = transform_y(y=y_test, num_date=num_date)

    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t

