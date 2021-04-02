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

# ============================================================
# =================creating DATA==============================
# ============================================================

def onepicmlp(sigma,amp1,bg , n, m):

    precision = 100
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)

    def gaussian_generator(mu, sigma, amp, bground):

        result = 0.5*(amp*((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv+1, mu, sigma))) + bground
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
    Mu = 40
    Amp = amp1
    sigma_training = sigma
    bground_training = np.zeros(precision) + bgconst
    numoftraning = 10000
    numofpredictions = 60000
    normalization = 2 * math.sqrt(bgconst)
    signalshape = gaussian_generator(Mu, sigma_training, Amp, bground_training)

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

    # data_testing_01 = []
    # data_testing_00 = []
    # label_testing = []
    # gaussian_test = gaussian_generator(Mu, sigma_training, Amp, bground_training)
    #
    # from1 = int(Mu - 1.5 * sigma_training)
    # to = int(Mu + 1.5 * sigma_training)
    #
    # manual_01 = []
    # manual_00 = []
    #
    # for _ in range(numofpredictions):
    #     aa, bb = fluctuation_manufacture(gaussian_test, bground_training)
    #     data_testing_01.append(aa)
    #     label_testing.append(1)
    #     aanormal = (np.array(aa) - bgconst) / normalization
    #     manual_01.append(np.trapz(aanormal[from1:to]))
    #
    # for _ in range(numofpredictions):
    #     aa, bb = fluctuation_manufacture(gaussian_test, bground_training)
    #     data_testing_00.append(bb)
    #     label_testing.append(0)
    #     bbnormal = (np.array(bb) - bgconst) / normalization
    #     manual_00.append(np.trapz(bbnormal[from1:to]))
    #
    # data_testing = data_testing_00 + data_testing_01
    # data_testing_01_norm = (np.array(data_testing_01) - bgconst) / normalization
    # data_testing_00_norm = (np.array(data_testing_00) - bgconst) / normalization
    # data_testing_norm = (np.array(data_testing) - bgconst) / normalization

    gaussian_test = gaussian_generator(Mu, sigma_training, Amp, bground_training)

    gausaray = []
    for _ in range(numofpredictions):
        gausaray.append(gaussian_test)

    aa, bb = fluctuation_manufacture(gausaray, bgconst)
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

    history = model.fit(np.array(data_combined_norm),
                        np.array(label_combined),
                        epochs=140,
                        validation_split=0.2,
                        verbose=1)
    print(model.summary())

    # -----------calculating the p value via ml----------

    median_00 = np.median(model.predict(data_testing_01_norm))
    sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00_norm)), median_00)

    plt.title('Histogram of Classifier Prediction')
    x00, t0 = np.histogram(model.predict(data_testing_00_norm), bins=100)
    x01, t1 = np.histogram(model.predict(data_testing_01_norm), bins=100)
    plt.plot(t0[0:-1], (x00/60000)*100, label='Without Signal')
    plt.plot(t1[0:-1], (x01/60000)*100, label='With signal')
    plt.legend(['Without Signal', 'With Signal'], loc='upper left')
    plt.ylabel('Probability Density')
    plt.xlabel('Value Of Prediction')
    plt.grid()
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Clasiffier Loss VS Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Validation', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

    print('\033[1m', 'ml calc', '\033[0m')
    print('#of values after median', sum_00)
    print('presantage', sum_00 / numofpredictions, '+-', math.sqrt(sum_00 / numofpredictions))

    print('run', n, 'out of', m)
    return sum_00 / numofpredictions

#  ------------------------------------------------------
#  -----------------BG-----------------------------------
# ------------------------------------------------------
#
# bg1222 = []
# pvalue1 = []
# pvalue2 = []
# pvalue3 = []
#
# bg = 100
# loops = 15
#
# for i in range(loops):
#      bg1222.append(bg)
#      pvalue2.append(onepicmlp(5, 80, bg, i+1, loops))
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
# df1.to_excel("1picmL.xlsx")
#  ------------------------------------------------------
#  -----------------AMP-------------------------------
pvalue2 = []
amp = 5
loops = 1
bg1222 =[]

for i in range(10):
     bg1222.append(amp)
     pvalue2.append(onepicmlp(5, 100, 130, i+1, loops))
     amp = amp + 20

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
df1.to_excel("1picClasiffier_amp.xlsx")