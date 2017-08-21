import numpy as np
import json
import glob

from utils import *

import pandas as pd
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.layers import Merge, Input, concatenate
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from keras import losses

from keras import backend as K
import seaborn as sns
sns.despine()

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys


STEP = 1
FORECAST = 24 * 60

data_original = pd.read_csv('./bitcoin-historical-data/USD_JPY_01_17.csv')[-20000:]
data_original = data_original.dropna()


def prepare_data(window):

    closep = data_original.ix[:, 'RATE'].tolist()[1:]

    volatility = pd.DataFrame(closep).rolling(window).std().values.tolist()
    volatility = [v[0] for v in volatility]

    X, Y = [], []
    for i in range(0, len(data_original), STEP):
        try:
            c = closep[i:i+window]
            volat = volatility[i:i+window]

            c = (np.array(c) - np.mean(c)) / np.std(c)
            volat = (np.array(volat) - np.mean(volat)) / np.std(volat)

            x_i = np.column_stack((c, volat))
            x_i = x_i.flatten()

            y_i = (closep[i+window+FORECAST] - closep[i+window]) / closep[i+window]

            if np.isnan(x_i).any():
                continue

        except Exception as e:
            break

        X.append(x_i)
        Y.append(y_i)

    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
    return X_train, X_test, Y_train, Y_test

# space = {'window': hp.choice('window',[30, 60, 120, 180]),
#         'units1': hp.choice('units1', [64, 512]),
#         'units2': hp.choice('units2', [64, 512]),
#         'units3': hp.choice('units3', [64, 512]),

#         'lr': hp.choice('lr',[0.01, 0.001, 0.0001]),
#         'activation': hp.choice('activation',['relu',
#                                                 'sigmoid',
#                                                 'tanh',
#                                                 'linear']),
#         'loss': hp.choice('loss', [losses.logcosh,
#                                     losses.mse,
#                                     losses.mae,
#                                     losses.mape])
#         }
# best: {'units1': 0, 'loss': 0, 'units3': 0, 'units2': 1, 'activation': 1, 'window': 1, 'lr': 0}
# best: {'units1': 0, 'loss': 2, 'units3': 1, 'units2': 0, 'activation': 0, 'window': 0, 'lr': 0}

X_train, X_test, Y_train, Y_test = prepare_data(72 * 60)

main_input = Input(shape=(len(X_train[0]), ), name='main_input')
x = Dense(512, activation='relu')(main_input)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

output = Dense(1, activation = "linear", name = "out")(x)
final_model = Model(inputs=[main_input], outputs=[output])
opt = Adam(lr=0.01)

final_model.compile(optimizer=opt, loss=losses.logcosh)

history = final_model.fit(X_train, Y_train,
          epochs = 50,
          batch_size = 256,
          verbose=1,
          validation_data=(X_test, Y_test),
          shuffle=True)


pred = final_model.predict(X_test)

predicted = pred
original = Y_test

plt.title('Actual and predicted')
plt.legend(loc='best')
plt.plot(original, color='black', label = 'Original data')
plt.plot(pred, color='blue', label = 'Predicted data')
plt.show()


print np.mean(np.square(predicted - original))
print np.mean(np.abs(predicted - original))
print np.mean(np.abs((original - predicted) / original))

# def experiment(params):
#     X_train, X_test, Y_train, Y_test = prepare_data(params['window'] * 60)

#     print 'Trying', params


#     try:
#         main_input = Input(shape=(len(X_train[0]), ), name='main_input')
#         x = Dense(params['units1'], activation=params['activation'])(main_input)
#         x = Dense(params['units2'], activation=params['activation'])(x)
#         x = Dense(params['units3'], activation=params['activation'])(x)

#         output = Dense(1, activation = "linear", name = "out")(x)
#         final_model = Model(inputs=[main_input], outputs=[output])
#         opt = Adam(lr=params['lr'])

#         final_model.compile(optimizer=opt,  loss=params['loss'])

#         history = final_model.fit(X_train, Y_train,
#                   epochs = 5,
#                   batch_size = 256,
#                   verbose=1,
#                   validation_data=(X_test, Y_test),
#                   shuffle=True)

#         pred = final_model.predict(X_test)

#     except:
#         print 'Something happened'
#         print '-' * 10

#         return {'loss': 999999, 'status': STATUS_OK}


#     predicted = pred
#     original = Y_test

#     mse = np.mean(np.square(predicted - original))

#     if np.isnan(mse):
#         print 'NaN happened'
#         print '-' * 10
#         return {'loss': 999999, 'status': STATUS_OK}

#     print mse
#     print '-' * 10

#     sys.stdout.flush()
#     return {'loss': mse, 'status': STATUS_OK}



# space = {'window': hp.choice('window',[24, 48, 72, 96]),
#         'units1': hp.choice('units1', [64, 512]),
#         'units2': hp.choice('units2', [64, 512]),
#         'units3': hp.choice('units3', [64, 512]),

#         'lr': hp.choice('lr',[0.01, 0.001, 0.0001]),
#         'activation': hp.choice('activation',['relu',
#                                                 'sigmoid',
#                                                 'tanh',
#                                                 'linear']),
#         'loss': hp.choice('loss', [losses.logcosh,
#                                     losses.mse,
#                                     losses.mae,
#                                     losses.mape])
#         }



# trials = Trials()
# best = fmin(experiment, space, algo=tpe.suggest, max_evals=50, trials=trials)
# print 'best: '
# print best

