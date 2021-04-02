
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers
from keras import optimizers
import pandas as pd
import random



def autoincodeP(signal, bg, amp):
    precision = 100
    yvalue = []
    for i in range(precision):
        yvalue.append(i)

    yvalue = np.array(yvalue)

    def totalloss (original, predicted):
        loss = (np.array(original) - np.array(predicted))**2
        loss = loss.sum(axis=1)
        return loss

    def gaussian_generator(mu, sigma, amp, bground):
        result = []
        result = 0.5 * (amp * ((stats.norm.pdf(yvalue, mu, sigma)) + stats.norm.pdf(yvalue + 1, mu, sigma))) + bground
        return result

    def sum_all_values_over(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i]>= number:
                value_returned += 1

        return value_returned

    def fluctuation_manufacture(numofevents_1, bground):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).size)
        data_bg = np.random.poisson(1 * bground, np.array(bground).size)

        return data_si_bg, data_bg

    # ============================================================
    # =================   creating TRAINING data   ===============
    # ============================================================

    bgconst = bg
    Mu = 40
    Amp = amp
    sigma_training = signal
    bground_training = np.zeros(precision) + bg
    epoch = 250
    normalization = 2 * math.sqrt(bgconst)
    numofpredictions = 10000

    signalshape = gaussian_generator(Mu, sigma_training, Amp, bground_training)
    no_sig_train = []
    with_sig = []
    no_sig_train = []
    # ==========================================================================================
    # ==============================  TRANING DATA FULL ARRAY  =================================
    # ==========================================================================================
    for i in range(60000):
        a, b = fluctuation_manufacture(gaussian_generator(random.uniform(10, 90), sigma_training, Amp, bground_training),
                                       bground_training)
        no_sig_train.append(b)
        with_sig.append(a)

    no_sig_train = np.array(no_sig_train)
    no_sig_train = (no_sig_train - bgconst) / normalization

    # ================= TEST DATA=========================



    no_sig_test = []
    sig_test = []

    for _ in range(numofpredictions):
        aa, bb = fluctuation_manufacture(gaussian_generator(random.uniform(10, 90), sigma_training, Amp, bground_training), bground_training)
        no_sig_test.append(bb)
        sig_test.append(aa)

    sig_test = np.array(sig_test)
    sig_test = (sig_test - bgconst) / normalization

    no_sig_test = np.array(no_sig_test)
    no_sig_test = (no_sig_test - bgconst) / normalization

    # # -----------------------------------------------------------------------
    # # --------------------------    MODEL     -------------------------------
    # # -----------------------------------------------------------------------

    input_data = Input(shape=(100,))
    encoded = Dense(60, activation='relu')(input_data)
    encoded = Dense(23, activation='relu' )(encoded)
    decoded = Dense(60, activation='relu' )(encoded)
    decoded = Dense(100, activation='linear')(decoded)

    autoencoder = Model(input_data, decoded)
    rmspop = optimizers.rmsprop(lr=0.001, rho=0.9)

    autoencoder.compile(loss="mse",
                        optimizer=rmspop,
                        metrics=['accuracy'])

    history = autoencoder.fit(no_sig_train, no_sig_train,
                              epochs=250,
                              shuffle=True,
                              validation_split=0.2,
                              verbose=1)

    print(autoencoder.summary())

    prdicted_nosig = autoencoder.predict(no_sig_test)
    predicted_sig = autoencoder.predict(sig_test)
    loss_nosig = totalloss(no_sig_test, prdicted_nosig)
    los_sig = totalloss(sig_test, predicted_sig)

    # -------------------- LOSS DISTREBUTION -----------------------

    # nosignal_data, nosignal_bin = np.histogram(np.array(loss_nosig), bins=np.arange(100))
    # signal_data, signal_bin = np.histogram(np.array(los_sig), bins=np.arange(100))
    # ---------------------- p-value calc ----------------------
    meadian = np.median(los_sig)
    print('meadian', meadian)

    sum1 = sum_all_values_over(np.array(loss_nosig), meadian)
    pvalue = sum1 / numofpredictions
    print(pvalue)

    return pvalue
##########################################################################

pvalue2 = []
amp = 50
loops = 10
bg1222 = []

for i in range(loops):

    bg1222.append(amp)
    pvalue2.append(autoincodeP(5, 200, amp))
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
df1.to_excel("autoincoder_amp_new_NEW.xlsx")
##################################################################
#################################################################
##################################################################

# pvalue2 = []
# bg = 50
# loops = 10
# bg1222 = []
#
# for i in range(loops):
#     bg1222.append(bg)
#     pvalue2.append(autoincodeP(5, bg, 80))
#     bg = bg + 20
#
#
# bg1222 = np.array(bg1222)
# pvalue2 = np.array(pvalue2)
#
# plt.title('Autoencoder Known Location')
# plt.ylabel('P-value')
# plt.yscale('log')
# plt.xlabel('Background Magnitude [Events/GeV]')
# plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
# plt.legend()
# plt.grid()
# plt.show()
#
# df1 = pd.DataFrame([bg1222, pvalue2],
#                    index=['bg', 'sigma5'])
# df1.to_excel("autoincode_bg2.xlsx")
