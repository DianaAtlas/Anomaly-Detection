

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers
import pandas as pd

# ============================================================
# =================    PARAMETERS    =========================
# ============================================================

def Bottlenackexp8(neurons):

    precision = 500
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)


    def gaussian_generator8(mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma, amp, exp):
        result = 0.5 * (amp * ((stats.norm.pdf(indexv, mu1, sigma)) + stats.norm.pdf(indexv + 1, mu1, sigma) +
                               (stats.norm.pdf(indexv, mu2, sigma)) + stats.norm.pdf(indexv + 1, mu2, sigma) +
                               (stats.norm.pdf(indexv, mu3, sigma)) + stats.norm.pdf(indexv + 1, mu3, sigma) +
                               (stats.norm.pdf(indexv, mu4, sigma)) + stats.norm.pdf(indexv + 1, mu4, sigma) +
                               (stats.norm.pdf(indexv, mu5, sigma)) + stats.norm.pdf(indexv + 1, mu5, sigma) +
                               (stats.norm.pdf(indexv, mu6, sigma)) + stats.norm.pdf(indexv + 1, mu6, sigma) +
                               (stats.norm.pdf(indexv, mu7, sigma)) + stats.norm.pdf(indexv + 1, mu7, sigma) +
                               (stats.norm.pdf(indexv, mu8, sigma)) + stats.norm.pdf(indexv + 1, mu8, sigma))) + exp

        return result


    def fluctuation_manufacture(numofevents_1, exp):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
        data_bg = np.random.poisson(1 * exp, np.array(numofevents_1).shape)

        return data_si_bg, data_bg


    def sum_all_values_over(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i]>= number:
                value_returned += 1

        return value_returned


    def totalloss(original, predicted):
            loss = (np.array(original) - np.array(predicted))
            loss = np.power(loss, 2)
            loss = loss.sum(axis=1)
            return loss
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
    amp = 10

    exp = 200 * np.exp(-indexv / 90)
    normalization = 2 * np.sqrt(exp)
    exp = np.array(exp)

    sigma_training = 5
    numofpredictions = 10000
    training = 40000
    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================

    no_sig_train = []
    gaussian_test = gaussian_generator8(Mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma_training, amp, exp)

    for _ in range(training):
        aa, bb = fluctuation_manufacture(gaussian_test, exp)
        no_sig_train.append(bb)

    no_sig_train = (np.array(no_sig_train) - np.array(exp)) / normalization


    # ============================================================
    # =================creating TEST data=========================
    # ============================================================


    expar = []
    normalizationar = []
    gausaray = []

    for _ in range(numofpredictions):
        gausaray.append(gaussian_test)
        expar.append(exp)
        normalizationar.append(normalization)


    sig_test, no_sig_test = fluctuation_manufacture(gausaray, exp)
    sig_test = (np.array(sig_test) - expar) / normalizationar
    no_sig_test = (np.array(no_sig_test) - expar) / normalizationar



    # # -----------------------------------------------------------------------
    # # --------------------------    MODEL     -------------------------------
    # # -----------------------------------------------------------------------

    input_data = Input(shape=(500,))
    encoded = Dense(300, activation='relu')(input_data)
    encoded = Dense(neurons, activation='relu')(encoded)
    decoded = Dense(300, activation='relu')(encoded)
    decoded = Dense(500, activation='linear')(decoded)

    autoencoder = Model(input_data, decoded)
    rmspop = optimizers.rmsprop(lr=0.0001, rho=0.9)

    autoencoder.compile(loss="mse",
                        optimizer=rmspop,
                        metrics=['accuracy'])

    history = autoencoder.fit(no_sig_train, no_sig_train,
                              epochs=40,
                              shuffle=True,
                              validation_split=0.2,
                              verbose=1)

    print(autoencoder.summary())


    # ---------------------- p-value calc ----------------------

    prdicted_nosig = autoencoder.predict(no_sig_test)
    predicted_sig = autoencoder.predict(sig_test)
    loss_nosig = totalloss(no_sig_test, prdicted_nosig)
    los_sig = totalloss(sig_test, predicted_sig)


    meadian = np.median(los_sig)
    print('meadian', meadian)
    sum1 = sum_all_values_over(np.array(loss_nosig), meadian)
    pvalue = sum1 / numofpredictions
    print(pvalue)


    return pvalue


###############################################################
bg1222 = []
pvalue2 = []

nuerons = 50
sigma2 = 5
loops = 16

for i in range(loops):
     bg1222.append(nuerons)
     pvalue2.append(Bottlenackexp8(nuerons))
     nuerons = nuerons + 15


bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)

print(pvalue2)
plt.title('8 exponential Bottleneck')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Neurons')
plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()


df1 = pd.DataFrame([bg1222, pvalue2 ],
                   index=['nuerons', 'pvalue'])
df1.to_excel("8picEX_Bottleneck_AMP.xlsx")