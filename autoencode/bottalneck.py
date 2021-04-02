

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
import matplotlib.patches as mpatches




def bottle(neurons,run,tottal):

    precision = 100
    yvalue = []
    for i in range(precision):
        yvalue.append(i)

    yvalue = np.array(yvalue)

    def gaussian_generator(mu, sigma, amp, bground):
        result = []
        result = 0.5 * (amp * ((stats.norm.pdf(yvalue, mu, sigma)) + stats.norm.pdf(yvalue + 1, mu, sigma))) + bground
        return result

    def fluctuation_manufacture(numofevents_1, bground):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).size)
        data_bg = np.random.poisson(1 * bground, np.array(bground).size)

        return data_si_bg, data_bg

    def totalloss (original, predicted):
            loss = (np.array(original) - np.array(predicted))
            loss = np.power(loss, 2)
            loss = loss.sum(axis=1)
            return loss

    def sum_all_values_over(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i]>= number:
                value_returned += 1

        return value_returned

    # ============================================================
    # =================   creating TRAINING data   ===============
    # ============================================================

    bgconst = 200
    Mu = 40
    Amp = 300
    variance_training = 25
    sigma_training = math.sqrt(variance_training)
    x = np.linspace(Mu - 10 * sigma_training, Mu + 10 * sigma_training, 100)
    bground_training = np.zeros(precision) + bgconst
    normalization = 2 * math.sqrt(bgconst)
    numofpredictions = 10000

    signalshape = gaussian_generator(Mu, sigma_training, Amp, bground_training)

    no_sig = []

    # ==========================================================================================
    # ==============================  TRANING DATA FULL ARRAY  =================================
    # ==========================================================================================
    for i in range(40000):
        a, b = fluctuation_manufacture(signalshape, bground_training)
        no_sig.append(b)

    no_sig = np.array(no_sig)
    no_sig = (no_sig-bgconst) / normalization

    # ================= TEST DATA=========================

    gaussian_test = gaussian_generator(Mu, sigma_training, Amp, bground_training)

    no_sig_test = []
    sig_test = []

    for _ in range(numofpredictions):
        aa, bb = fluctuation_manufacture(gaussian_test, bground_training)
        no_sig_test.append(bb)
        sig_test.append(aa)

    sig_test = np.array(sig_test)
    sig_test = (sig_test - bgconst) / normalization

    no_sig_test = np.array(no_sig_test)
    no_sig_test = (no_sig_test - bgconst) / normalization

    print('No SIGNAL TEST', no_sig_test)
    print('SIGNAL TEST', sig_test)

    # # -----------------------------------------------------------------------
    # # --------------------------    MODEL     -------------------------------
    # # -----------------------------------------------------------------------

    input_data = Input(shape=(100,))
    encoded = Dense(60, activation='relu')(input_data)
    encoded = Dense(neurons, activation='relu')(encoded)
    decoded = Dense(60, activation='relu')(encoded)
    decoded = Dense(100, activation='linear')(decoded)

    autoencoder = Model(input_data, decoded)
    rmspop = optimizers.rmsprop(lr=0.001, rho=0.9)

    autoencoder.compile(loss="mse",
                        optimizer=rmspop,
                        metrics=['accuracy'])

    history = autoencoder.fit(no_sig, no_sig,
                              epochs=250,
                              shuffle=True,
                              validation_split=0.2,
                              verbose=1)

    print(autoencoder.summary())

    # ---------------------- p-value calc ----------------------

    prdicted_nosig = autoencoder.predict(no_sig_test)
    predicted_sig = autoencoder.predict(sig_test)
    loss_nosig = totalloss(no_sig_test, prdicted_nosig)
    los_sig = totalloss(sig_test, predicted_sig)


    print('run', run, 'out of', tottal)
    meadian = np.median(los_sig)
    print('meadian', meadian)
    sum1 = sum_all_values_over(np.array(loss_nosig), meadian)
    pvalue = sum1 / numofpredictions
    print('p-value is', pvalue)

    return pvalue
##########################################################################
##################################################################
#################################################################
##################################################################


pvalue2 = []
neurons = 5
loops = 10
bg1222 = []

for i in range(loops):
    bg1222.append(neurons)
    pvalue2.append(bottle(neurons, i, loops))
    neurons = neurons + 5


bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)

plt.title('Autoencoder Known Location nuerons ')
plt.ylabel('P-value')
plt.xlabel('nuerons ')
plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue2],
                   index=['nuerons', 'sigma5'])
df1.to_excel("autoincode_nuerons2.xlsx")



