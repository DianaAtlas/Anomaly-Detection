import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras import optimizers
import random
from keras.layers import Dense, Input
from keras.models import Model
import pandas as pd


def MassAutoencoder(mu):
    precision = 100
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)

    def fluctuation_manufacture(sigma, amp, bground, mu):

        signalshape = 0.5 * (amp * ((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv + 1, mu, sigma))) + bground
        np.array(signalshape)
        data_si_bg = np.random.poisson(signalshape, np.array(signalshape).size)
        data_bg = np.random.poisson(1 * bground, np.array(bground).size)

        return data_si_bg, data_bg

    def totalloss(original, predicted):
            loss = (np.array(original) - np.array(predicted))
            loss = np.power(loss, 2)
            loss = loss.sum(axis=1)
            return loss

    def sum_all_values_over(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i] >= number:
                value_returned += 1

        return value_returned

    # ============================================================
    # =================  PARAMETER SETTINGS   ====================
    # ============================================================

    exp = 200 * np.exp(-indexv / 90)
    Amp = 150
    sigma_training = 5
    bground_training = np.zeros(precision) + exp
    numoftraning = 50000
    numofpredictions = 60000
    normalization = 2 * np.sqrt(exp)

    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================

    nosig_training = []
    for _ in range(numoftraning):
        a, b = fluctuation_manufacture(sigma_training, Amp, bground_training, mu)
        nosig_training.append(b)

    nosig_training = (np.array(nosig_training) - exp) / normalization

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    sig_test = []
    nosig_test = []

    for _ in range(numofpredictions):
        aa, bb = fluctuation_manufacture(sigma_training, Amp, bground_training, mu)
        sig_test.append(aa)
        nosig_test.append(bb)

    sig_test = (np.array(sig_test) - exp) / normalization
    nosig_test = (np.array(nosig_test) - exp) / normalization

    # -------------------------------------------------------------
    # ------        MODEL    --------------------------------------
    # -------------------------------------------------------------

    input_data = Input(shape=(100,))
    encoded = Dense(60, activation='relu')(input_data)
    encoded = Dense(15, activation='relu')(encoded)
    decoded = Dense(60, activation='relu')(encoded)
    decoded = Dense(100, activation='linear')(decoded)

    autoencoder = Model(input_data, decoded)
    rmspop = optimizers.rmsprop(lr=0.0001, rho=0.9)

    autoencoder.compile(loss="mse",
                        optimizer=rmspop,
                        metrics=['accuracy'])

    history = autoencoder.fit(nosig_training, nosig_training,
                              epochs=130,
                              shuffle=True,
                              validation_split=0.2,
                              verbose=1)

    print(autoencoder.summary())

    # -------------------------------------------------------------
    # ------        P-VALUE CALC    -------------------------------
    # -------------------------------------------------------------

    prdicted_nosig = autoencoder.predict(nosig_test)
    predicted_sig = autoencoder.predict(sig_test)
    loss_nosig = totalloss(nosig_test, prdicted_nosig)
    los_sig = totalloss(sig_test, predicted_sig)

    meadian = np.median(los_sig)
    print('meadian', meadian)

    sum1 = sum_all_values_over(np.array(loss_nosig), meadian)
    pvalue = sum1 / numofpredictions
    print('mass', mu)
    print('pvalue', pvalue)

    return pvalue

# -------------------------------------------------------------
# ------------              AMP            --------------------
# -------------------------------------------------------------

pvalue2 = []
mass1 = 10
loops = 10
bg1222 = []

for i in range(loops):
    bg1222.append(mass1)
    pvalue2.append(MassAutoencoder(mass1))
    mass1 = mass1 + 10


bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)

plt.title('Autoencoder Unknown location')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Signal Magnitude [Events/GeV]')
plt.plot(bg1222, pvalue2, 'o', label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue2],
                   index=['amp', 'sigma5'])
df1.to_excel("Mass Autoencoder15.xlsx")

