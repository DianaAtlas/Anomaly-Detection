
import matplotlib.pyplot as plt
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

precision = 100
indexv = []
for i in range(precision):
    indexv.append(i)

indexv = np.array(indexv)


def gaussian_generator(mu, sigma, amp, exp):
    result = []
    result = 0.5*(amp*((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv+1, mu, sigma))) + exp
    return result


def fluctuation_manufacture(numofevents_1, array):
    np.array(numofevents_1)
    data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).size)
    data_bg = np.random.poisson(array, np.array(array).size)

    return data_si_bg, data_bg


def sum_all_values_over(array, number):
    value_returned = 0
    for i in range(len(array)):
        if array[i][0] >= number:
            value_returned += 1

    return value_returned


def sum_all_values_under(array, number):
    value_returned = 0
    for i in range(len(array)):
        if array[i][0] <= number:
            value_returned += 1

    return value_returned


def sum_all_values_over_manual(array, number):
    value_returned = 0
    for i in range(len(array)):
        if array[i] >= number:
            value_returned += 1

    return value_returned
# ============================================================
# =================creating TRAINING data=====================
# ============================================================

def MClassifier(mu):
    Mu = mu
    Amp = 130
    variance_training = 25
    sigma_training = math.sqrt(variance_training)
    numoftraning = 10000
    numofpredictions = 60000

    exp = 130 * np.exp(-indexv / 100000000)
    signalshape = gaussian_generator(40, sigma_training, Amp, exp)

    normalization = 2 * np.sqrt(exp)

    data_combined = []
    label_combined = []

    for _ in range(numoftraning):
        a, b = fluctuation_manufacture(signalshape, exp)
        data_combined.append((a - exp) / normalization)
        label_combined.append(1)
        data_combined.append((b - exp) / normalization)
        label_combined.append(0)

    data_combined_norm = np.array(data_combined)

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    data_testing_01 = []
    data_testing_00 = []

    for _ in range(numofpredictions):
        aa, bb = fluctuation_manufacture(signalshape, exp)
        data_testing_01.append((aa - exp) / normalization)
        data_testing_00.append((bb - exp) / normalization)

    data_testing_01_norm = np.array(data_testing_01)
    data_testing_00_norm = np.array(data_testing_00)

    # -----------------------------------------------------------------------
    # ------ MODEL---------------------------------------------------
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

    # print('NN predicted', model.predict([np.array(data_testing_norm)]))
    # print((lable_testing))
    # plt.plot(model.predict([np.array(data_testing_norm)]))
    # plt.show()
    # print(model.predict(np.array(data_testing_00_norm)))
    # plt.plot(model.predict(np.array(data_testing_00_norm)))
    # plt.show()

    # ===================Getting the weights============
    W_0 = model.layers[0].get_weights()[0]
    plt.plot(W_0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('M[GeV]', fontsize = 15)
    plt.ylabel('weight', fontsize = 15)
    plt.title('Weights Layer 1', fontsize =20)
    plt.grid()
    plt.show()

    # ====================Histogram=============
    # plt.xlabel('value', color='#1C2833')
    # plt.ylabel('# of times counted', color='#1C2833')
    #
    # plt.title('Histogram')
    #
    # y, bi = np.histogram(model.predict(np.array(data_testing_01_norm)), bins=np.arange(50+1)/50)
    # y2, bi2 = np.histogram(model.predict(np.array(data_testing_00_norm)), bins=np.arange(50+1)/50)
    #
    # plt.plot(bi[0:-1], y)
    # plt.plot(bi2[0:-1], y2)
    #
    # plt.show()
    #
    # difference = 1-y
    # t = 1
    # plt.show()


    # # ================ summarize history for loss

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Classifier Exponential Background')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid()
    plt.show()

    # -----------calculating the p value via ml----------

    median_00 = np.median(model.predict(data_testing_01_norm))
    sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00_norm)), median_00)

    print('\033[1m', 'ml calc', '\033[0m')
    print('#of values after median', sum_00)
    print('presantage', sum_00 / numofpredictions, '+-', math.sqrt(sum_00) / numofpredictions)
    pvalue = sum_00 / numofpredictions

    return pvalue

# ======================================================
# =================PVALUE VS MASS  =====================
# ======================================================

#
mass =[]
pvalue = []

mass1 = 10
sigma2 = 5
loops = 10

for i in range(loops):
     mass.append(mass1)
     pvalue.append(MClassifier(40))
     mass1 = mass1 + 10


mass = np.array(mass)
pvalue = np.array(pvalue)

plt.title('P-value VS Background')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Background')
plt.plot(mass, pvalue, 'o', label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()


df1 = pd.DataFrame([mass, pvalue ],
                   index=['mass', 'sigma5'])
df1.to_excel("8ClssifierBG.xlsx")


