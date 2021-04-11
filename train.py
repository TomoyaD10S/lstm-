from preprocess import preprocess
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import metrics
from matplotlib import pyplot as plt

X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = preprocess()

print(X_train_t.shape)

num_l1 = 300
num_l2 = 100
num_output = 1

model = Sequential()

model.add(LSTM(units=num_l1,
                activation='tanh',
                input_shape=(X_train_t.shape[1], X_train_t.shape[2]),
                recurrent_activation='hard_sigmoid'))

model.add(Dense(num_l2))

model.add(Dense(num_output))

model.summary()

model.compile(loss='MAPE', optimizer='adam', metrics=[metrics.mae])

result = model.fit(x=X_train_t, y=y_train_t, epochs=100, batch_size=64, validation_data=(X_val_t, y_val_t))

model.save('./model/lstm_model.h5')
model.save_weights('./weights/lstm_weights.h5')

plt.plot(result.history['mean_absolute_error'])
plt.plot(result.history['val_mean_absolute_error'])
plt.legend(['Train', 'Val'])
plt.xlabel('Epoch')
plt.ylabel('MAE')

plt.show()



