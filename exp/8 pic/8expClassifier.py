

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
def mlpveight(amp1, bg):
    precision = 500
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)


    def gaussian_generator8(mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma, amp, exp):
        result = []
        result = 0.5*(amp*((stats.norm.pdf(indexv, mu1, sigma)) + stats.norm.pdf(indexv+1, mu1, sigma) +
                           (stats.norm.pdf(indexv, mu2, sigma)) + stats.norm.pdf(indexv+1, mu2, sigma) +
                           (stats.norm.pdf(indexv, mu3, sigma)) + stats.norm.pdf(indexv+1, mu3, sigma) +
                           (stats.norm.pdf(indexv, mu4, sigma)) + stats.norm.pdf(indexv+1, mu4, sigma) +
                           (stats.norm.pdf(indexv, mu5, sigma)) + stats.norm.pdf(indexv+1, mu5, sigma) +
                           (stats.norm.pdf(indexv, mu6, sigma)) + stats.norm.pdf(indexv+1, mu6, sigma) +
                           (stats.norm.pdf(indexv, mu7, sigma)) + stats.norm.pdf(indexv+1, mu7, sigma) +
                           (stats.norm.pdf(indexv, mu8, sigma)) + stats.norm.pdf(indexv+1, mu8, sigma))) + exp

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

    mu2 = 30
    mu3 = 90
    Mu1 = 150
    mu4 = 210
    mu5 = 270
    mu6 = 330
    mu7 = 390
    mu8 = 450

    sigma_training = 5

    exp = bg * np.exp(-indexv / 90)
    exp = np.array(exp)

    normalization = 2 * np.sqrt(exp)
    numoftraning = 10000
    numofpredictions = 60000

    signalshape = gaussian_generator8(Mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma_training, amp1, exp)

    # ============================================================
    # =================creating TRAIN data========================
    # ============================================================
    data_combined = []
    label_combined = []
    exp_train = []
    normalization_train = []

    for _ in range(numoftraning):
        a, b = fluctuation_manufacture(signalshape, exp)
        data_combined.append(a)

        label_combined.append(1)
        data_combined.append(b)
        label_combined.append(0)
        exp_train.append(exp)
        exp_train.append(exp)
        normalization_train.append(normalization)
        normalization_train.append(normalization)

    data_combined_norm = (np.array(data_combined) - exp_train) / normalization_train

    print('data combined norm', np.array(data_combined_norm).shape, data_combined_norm)

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================
    data_testing_01 = []
    data_testing_00 = []
    exp_test = []
    normalization_test = []

    for _ in range(numofpredictions):
        exp_test.append(exp)
        normalization_test.append(normalization)
        a, b = fluctuation_manufacture(signalshape, exp)
        data_testing_01.append(a)
        data_testing_00.append(b)

    data_testing_01 = (np.array(data_testing_01) - exp_test) / normalization_test
    data_testing_00 = (np.array(data_testing_00) - exp_test) / normalization_test

    # -----------------------------------------------------------------------
    # -------------- MODEL---------------------------------------------------
    # -----------------------------------------------------------------------

    model = Sequential()

    model.add(Dense(24, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.005)))  # Kernel Regularization regulates the weights

    model.add(Dense(32, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.005)))

    model.add(Dense(12, batch_size=500,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.005)))

    model.add(Dense(1, activation='sigmoid'))

    rmspop = optimizers.rmsprop(lr=0.003, rho=0.9)

    model.compile(loss='binary_crossentropy',
                  optimizer=rmspop,
                  metrics=['accuracy'])

    history = model.fit(np.array(data_combined_norm),
                        np.array(label_combined),
                        epochs=140,
                        validation_split=0.2,
                        verbose=1)
    print(model.summary())

    # -----------calculating the p value via ml----------

    median_00 = np.median(model.predict(data_testing_01))
    sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00)), median_00)

    print('\033[1m', 'ml calc', '\033[0m')
    print('#of values after median', sum_00)
    print('presantage', sum_00 / numofpredictions, '+-', math.sqrt(sum_00 / numofpredictions))
    print('bg', bg)

    # -------------------- LOSS DISTREBUTION -----------------------
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')

    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')





    pvalue = sum_00 / numofpredictions
    return pvalue

#  ------------------------------------------------------
#  -----------------BG-----------------------------------
# ------------------------------------------------------

bg1222 = []
pvalue2 = []

bg = 300
loops = 10

for i in range(loops):
     bg1222.append(bg)
     pvalue2.append(mlpveight(9, bg))
     bg = bg + 7.5


bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)

print(pvalue2)
print(bg1222)
plt.title('P-value VS Background')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Background')
plt.plot(bg1222, pvalue2, 'o', label='Classifier')
plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue2 ],
                   index=['bg', 'pvalue'])
df1.to_excel("8PIC_CLASSIFIER_BGggggg.xlsx")


#  ------------------------------------------------------
#  -----------------amp-----------------------------------
# ------------------------------------------------------

# bg1222 = []
# pvalue2 = []
#
# amp = 0
# sigma2 = 5
# loops = 10
#
# for i in range(loops):
#      bg1222.append(amp)
#      pvalue2.append(mlpveight(amp, 200))
#      amp = amp + 3
#
# bg1222 = np.array(bg1222)
# pvalue2 = np.array(pvalue2)
#
# print('pvalue2', pvalue2)
# print('bg1222', bg1222)
# plt.title('8 exponential Classifier')
# plt.ylabel('P-value')
# plt.yscale('log')
# plt.xlabel('Signal Magnitud')
# plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
# plt.legend()
# plt.grid()
# plt.show()
#
#
# df1 = pd.DataFrame([bg1222, pvalue2],
#                    index=['amp', 'sigma5'])
# df1.to_excel("8picEX_CLASIFFIER1_AMP.xlsx")
