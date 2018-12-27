import keras
import matplotlib.pyplot as plt
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from sklearn import datasets
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target

X_train, X_test, y_train, y_test = train_test_split(
    df[['RM', 'LSTAT', 'PTRATIO']], df[['target']], test_size=0.3, random_state=0)
X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)
y_train = MinMaxScaler().fit_transform(y_train)
y_test = MinMaxScaler().fit_transform(y_test)

m = len(X_train)
n = 3  # number of features
n_hidden = 20  # number of hidden neurons
batch_size = 200
eta = 0.01
max_epoch = 1000

# build model
model = Sequential()
model.add(Dense(n_hidden, input_dim=n, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=max_epoch,
    batch_size=batch_size,
    verbose=1,
)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
r2 = r2_score(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred)
print(f'Performance Metrics: R2 {r2}, RMSE {rmse}')
