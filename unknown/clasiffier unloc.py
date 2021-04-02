import matplotlib.pyplot as plt
import pandas as pd
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
import random


# ============================================================
# =================creating DATA==============================
# ============================================================
def unmlp(sigma, bg, amp):
    precision = 100
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)

    def fluctuation_manufacturetrain(sigma, amp, bground):
        mu = random.uniform(10, 90)
        signalshape = 0.5 * (
                    amp * ((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv + 1, mu, sigma))) + bground
        np.array(signalshape)
        data_si_bg = np.random.poisson(signalshape, np.array(signalshape).size)
        data_bg = np.random.poisson(1 * bground, np.array(bground).size)

        return data_si_bg, data_bg

    def sum_all_values_over(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i][0] >= number:
                value_returned += 1

        return value_returned

    def gaussiancreator(amp, bground,sigma):
        mu = random.uniform(10, 90)
        signalshape = 0.5 * (
                    amp * ((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv + 1, mu, sigma))) + bground
        return signalshape

    def fluctuation_manufacture(numofevents_1, bground):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
        data_bg = np.random.poisson(1 * bground, np.array(numofevents_1).shape)

        return data_si_bg, data_bg

    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================

    bgconst = bg
    Amp = amp
    sigma_training = sigma
    bground_training = np.zeros(precision) + bgconst
    numoftraning = 20000
    numofpredictions = 60000
    normalization = 2 * math.sqrt(bgconst)

    combinedtrain = []
    labeltrain = []

    for _ in range(numoftraning):
        aa, bb = fluctuation_manufacturetrain(sigma, amp, bground_training)
        combinedtrain.append((np.array(aa) - bgconst) / normalization)
        labeltrain.append(1)
        combinedtrain.append((np.array(bb) - bgconst) / normalization)
        labeltrain.append(0)

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    gausary = []
    for _ in range(numofpredictions):
        gausary.append(gaussiancreator(amp, bg, sigma))

    aa, bb = fluctuation_manufacture(gausary, bgconst)

    data_testing_01_norm = (np.array(aa) - bgconst) / normalization
    data_testing_00_norm = (np.array(bb) - bgconst) / normalization

    # -----------------------------------------------------------------------
    # ------ 2 model layer---------------------------------------------------
    # -----------------------------------------------------------------------

    model = Sequential()

    model.add(Dense(24, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.003)))  # Kernel Regularization regulates the weights

    model.add(Dense(32, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.003)))

    model.add(Dense(12, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.003)))

    model.add(Dense(1, activation='sigmoid'))

    rmspop = optimizers.rmsprop(lr=0.003, rho=0.9)

    model.compile(loss='binary_crossentropy',
                  optimizer=rmspop,
                  metrics=['accuracy'])

    history = model.fit(np.array(combinedtrain),
                        np.array(labeltrain),
                        epochs=180,
                        validation_split=0.2,
                        verbose=1)
    print(model.summary())

    # -----------calculating the p value via ml----------

    median_00 = np.median(model.predict(data_testing_01_norm))
    sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00_norm)), median_00)

    print('\033[1m', 'ml calc', '\033[0m')
    print('for bg of', bg)
    print('presantage', sum_00 / numofpredictions, '+-', math.sqrt(sum_00 / numofpredictions))
    pvalue = sum_00 / numofpredictions

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Classifier Unknown Location')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid()
    plt.show()

    return pvalue


#  ------------------------------------------------------
#  -----------------BG-----------------------------------
# ------------------------------------------------------
# bg1222 = []
# pvalue2 = []
#
#
# bg = 150
# loops = 10
#
# for i in range(loops):
#      bg1222.append(bg)
#      pvalue2.append(unmlp(5, bg, 80))
#      bg = bg + 15
#
# bg1222 = np.array(bg1222)
# pvalue2 = np.array(pvalue2)
#
#
# plt.title('P-value VS Background')
# plt.ylabel('P-value')
# plt.yscale('log')
# plt.xlabel('Background')
# plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
# plt.legend()
# plt.grid()
# plt.show()
#
#
# df1 = pd.DataFrame([bg1222, pvalue2],
#                    index=['bg', 'sigma5'])
# df1.to_excel("pvml_bg_allllll.xlsx")
#  ------------------------------------------------------
#  -----------------AMP-------------------------------


pvalue2 = []
amp = 150
loops = 5
bg1222 = []

for i in range(loops):
    bg1222.append(amp)
    pvalue2.append(unmlp(5, 200, amp))
    amp = amp + 10


bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)

plt.title('P-value VS Signal Magnitude')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Signal Magnitude [Events/GeV]')
plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue2],
                   index=['amp', 'sigma5'])
df1.to_excel("unkownCLASIFFIER-amplow33333333333333333.xlsx")