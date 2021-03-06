

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import scipy.stats as stats
import math
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import regularizers
from keras import optimizers
import pandas as pd


# ============================================================
# =================creating DATA==============================
# ============================================================
def mlpveight(sig, amp1, bg,n,m):
    precision = 500
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)


    def gaussian_generator8(mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma, amp, bground):
        result = []
        result = 0.5*(amp*((stats.norm.pdf(indexv, mu1, sigma)) + stats.norm.pdf(indexv+1, mu1, sigma) +
                           (stats.norm.pdf(indexv, mu2, sigma)) + stats.norm.pdf(indexv+1, mu2, sigma) +
                           (stats.norm.pdf(indexv, mu3, sigma)) + stats.norm.pdf(indexv+1, mu3, sigma) +
                           (stats.norm.pdf(indexv, mu4, sigma)) + stats.norm.pdf(indexv+1, mu4, sigma) +
                           (stats.norm.pdf(indexv, mu5, sigma)) + stats.norm.pdf(indexv+1, mu5, sigma) +
                           (stats.norm.pdf(indexv, mu6, sigma)) + stats.norm.pdf(indexv+1, mu6, sigma) +
                           (stats.norm.pdf(indexv, mu7, sigma)) + stats.norm.pdf(indexv+1, mu7, sigma) +
                           (stats.norm.pdf(indexv, mu8, sigma)) + stats.norm.pdf(indexv+1, mu8, sigma))) + bground

        return result

    def fluctuation_manufacture(numofevents_1, bground):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
        data_bg = np.random.poisson(1 * bground, np.array(numofevents_1).shape)

        return data_si_bg, data_bg

    def sum_all_values_over(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i][0] >= number:
                value_returned += 1

        return value_returned

    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================

    bgconst = bg

    mu2 = 30
    mu3 = 90
    Mu1 = 150
    mu4 = 210
    mu5 = 270
    mu6 = 330
    mu7 = 390
    mu8 = 450

    Amp = amp1
    sigma_training = sig

    bground_training = np.zeros(precision) + bgconst
    numoftraning = 10000
    numofpredictions = 60000
    epoch = 100
    normalization = 2 * math.sqrt(bgconst)
    signalshape = gaussian_generator8(Mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma_training, Amp, bground_training)


    data_combined = []
    label_combined = []

    for _ in range(numoftraning):
        a, b = fluctuation_manufacture(signalshape, bground_training)
        data_combined.append(a)
        label_combined.append(1)
        data_combined.append(b)
        label_combined.append(0)

    data_combined_norm = (np.array(data_combined) - bgconst) / normalization


    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    gaussian_test = gaussian_generator8(Mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma_training, Amp, bground_training)

    gausaray = []
    for _ in range(numofpredictions):
        gausaray.append(gaussian_test)

    aa, bb = fluctuation_manufacture(gausaray, bground_training)

    data_testing_01_norm = (np.array(aa) - bgconst) / normalization
    data_testing_00_norm = (np.array(bb) - bgconst) / normalization

    # -----------------------------------------------------------------------
    # ------ 2 model layer---------------------------------------------------
    # -----------------------------------------------------------------------

    model = Sequential()
    model.add(Dense(24, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dense(32, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dense(12, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dense(1, activation='sigmoid'))

    rmspop = optimizers.rmsprop(lr=0.01, rho=0.9)

    model.compile(loss='binary_crossentropy',
                  optimizer=rmspop,
                  metrics=['accuracy'])

    history = model.fit(np.array(data_combined_norm),
                        np.array(label_combined),
                        epochs=350,
                        validation_split=0.2,
                        verbose=1,
                        shuffle=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')

    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    #-----------calculating the p value via ml----------

    median_00 = np.median(model.predict(data_testing_01_norm))
    sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00_norm)), median_00)

    print ('\033[1m', 'ml calc', '\033[0m')
    print('#of values after median', sum_00)
    print('presantage', sum_00 / numofpredictions, '+-', math.sqrt(sum_00 / numofpredictions))
    print('run', n, 'out of', m)

    pvalue = sum_00 / numofpredictions
    return pvalue

#  ------------------------------------------------------
#  -----------------BG-----------------------------------
# ------------------------------------------------------

bg1222 = []
pvalue2 = []

bg = 300
sigma1 = 3
sigma2 = 5
sigma3 = 10
loops = 5

for i in range(loops):
     bg1222.append(bg)
     pvalue2.append(mlpveight(5, 30, bg, i+1, loops))
     bg = bg + 20


bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)

print(pvalue2)
plt.title('P-value VS Background')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Background')
plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()


df1 = pd.DataFrame([bg1222, pvalue2 ],
                   index=['bg', 'sigma5'])
df1.to_excel("8PIC_ML_BG.xlsx")


#  ------------------------------------------------------
#  -----------------amp-----------------------------------
# ------------------------------------------------------
#
# bg1222 = []
# pvalue2 = []
#
# amp = 25
# sigma1 = 3
# sigma2 = 5
# sigma3 = 10
# loops = 10
#
# for i in range(loops):
#      bg1222.append(bg)
#      pvalue2.append(mlpveight(5, amp, 200, i+1, loops))
#      bg = amp + 5
#
#
# bg1222 = np.array(bg1222)
# pvalue2 = np.array(pvalue2)
#
# print(pvalue2)
# plt.title('P-value VS Signal Magnitude')
# plt.ylabel('P-value')
# plt.yscale('log')
# plt.xlabel('Background')
# plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
# plt.legend()
# plt.grid()
# plt.show()
#
#
# df1 = pd.DataFrame([bg1222, pvalue2 ],
#                    index=['amp', 'sigma5'])
# df1.to_excel("8picampml.xlsx")
