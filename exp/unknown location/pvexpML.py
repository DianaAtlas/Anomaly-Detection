import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import optimizers
import pandas as pd
import random



def expmlp(sigma,amp1,bg, n, m):

    precision = 100
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)

    def gaussian_generator(mu, sigma, amp, exp):

        result = 0.5*(amp*((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv+1, mu, sigma))) + exp
        return result

    def fluctuation_manufacture(numofevents_1, exp):
            np.array(numofevents_1)
            data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
            data_bg = np.random.poisson(1 * exp, np.array(numofevents_1).shape)

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


    Amp = amp1
    sigma_training = sigma
    numoftraning = 10000
    numofpredictions = 60000
    exp = bg * np.exp(-indexv / 90)
    exp = np.array(exp)
    normalization = 2 * np.sqrt(exp)


    data_combined = []
    label_combined = []

    normtraning = []
    exptraning = []

    for _ in range(numoftraning):
        a, b = fluctuation_manufacture(gaussian_generator(random.uniform(10, 90), sigma_training, Amp, exp)
                                       , exp)
        data_combined.append(a)
        label_combined.append(1)
        data_combined.append(b)
        label_combined.append(0)
        normtraning.append(normalization)
        normtraning.append(normalization)
        exptraning.append(exp)
        exptraning.append(exp)

    data_combined_norm = (np.array(data_combined) - exptraning) / normtraning

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================



    expar =[]
    normalizationar = []
    gausaray = []

    for _ in range(numofpredictions):
        gausaray.append(gaussian_generator(random.uniform(10, 90), 5, amp1, exp))
        expar.append(exp)
        normalizationar.append(normalization)

    aa, bb = fluctuation_manufacture(gausaray, exp)
    data_testing_01_norm = (np.array(aa) - expar) / normalizationar
    data_testing_00_norm = (np.array(bb) - expar) / normalizationar

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

    # -----------calculating the p value via ml----------

    median_00 = np.median(model.predict(data_testing_01_norm))
    sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00_norm)), median_00)

    print('\033[1m', 'ml calc', '\033[0m')
    print('#of values after median', sum_00)
    print('presantage', sum_00 / numofpredictions, '+-', math.sqrt(sum_00 / numofpredictions))

    print('run', n, 'out of', m)
    return sum_00 / numofpredictions


#  ----------------   MU  -----------------------------

# mu = 5
# loops = 10
# bg1222 = []
# pvalue2 = []
#
# for i in range(loops):
#      bg1222.append(mu)
#      pvalue2.append(expmlp(5, 80, mu, i+1, loops))
#      mu = mu + 10
#
# bg1222 = np.array(bg1222)
# pvalue2 = np.array(pvalue2)
#
# plt.title('P-value VS Pic Location ML')
# plt.ylabel('P-value')
# plt.yscale('log')
# plt.xlabel('mass [GeV]')
# plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
# plt.legend()
# plt.grid()
# plt.show()
#
# df1 = pd.DataFrame([bg1222, pvalue2],
#                    index=['mu', 'sigma5'])
# df1.to_excel("1picexpML.xlsx")

# ----------------------- AMP-------------------


bg1222 = []
pvalue2 = []


amp = 5
loops = 10
for i in range(loops):
     bg1222.append(amp)
     pvalue2.append(expmlp(5, amp,200,  i+1, loops))
     amp = amp + 19

bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)

plt.title('unknown location and exponent')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('amp [GeV]')
plt.plot(bg1222, pvalue2, 'o', label='Classifier')
plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue2],
                   index=['amp', 'pvalue'])
df1.to_excel("Unknown_Exp_cLASSIFIERNEW.xlsx")