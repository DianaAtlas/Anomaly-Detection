

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
import random
import matplotlib.patches as mpatches

# ============================================================
# =================    PARAMETERS    =========================
# ============================================================

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

Amp = 500
variance_training = 25
sigma_training = math.sqrt(variance_training)
bground_training = np.zeros(precision) + bgconst
normalization = 2 * math.sqrt(bgconst)
numofpredictions = 10000


no_sig = []
no_sig_test = []
sig_test = []

# ==========================================================================================
# ==============================  TRANING DATA FULL ARRAY  =================================
# ==========================================================================================
for i in range(40000):
    a, b = fluctuation_manufacture(gaussian_generator(random.uniform(10,90), sigma_training, Amp, bground_training),
                                   bground_training)
    no_sig.append(b)

no_sig = np.array(no_sig)
no_sig = (no_sig-bgconst) / normalization

# ================= TEST DATA=========================


no_sig_test = []
sig_test = []

for _ in range(numofpredictions):
    aa, bb = fluctuation_manufacture(gaussian_generator(random.uniform(10,90), sigma_training, Amp, bground_training),
                                     bground_training)
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
encoded = Dense(60, activation='relu' )(input_data)
encoded = Dense(20, activation='relu' )(encoded)
decoded = Dense(60, activation='relu' )(encoded)
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

# plt.plot(loss_nosig, label='No signal')
# plt.plot(los_sig, label='Signal')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
# plt.show()

meadian = np.median(los_sig)
print('meadian', meadian)
sum1 = sum_all_values_over(np.array(loss_nosig), meadian)
pvalue = sum1 / numofpredictions
print(pvalue)


plt.title('Autoencoder No signal prediction')
plt.plot(prdicted_nosig[0], label='predicted signal')
plt.plot(no_sig_test[0], label='original signal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
plt.show()


plt.title('Autoencoder signal prediction')
plt.plot(predicted_sig[0], label='predicted signal')
plt.plot(sig_test[0], label='original signal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
plt.show()

#-------------------- LOSS DISTREBUTION -----------------------
plt.xlabel('value', color='#1C2833')
plt.ylabel('Loss Distribution', color='#1C2833')
plt.title('Loss Distribution')

y, bi = np.histogram(np.array(loss_nosig), bins=np.arange(100))
y2, bi2 = np.histogram(np.array(los_sig), bins=np.arange(100))
print('y', y.shape, y)
print('y2', y2.shape, y2)

plt.plot(bi[0:-1], y, label='No Signal')
plt.plot(bi2[0:-1], y2, label='Signal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
plt.show()

difference = 1-y
t = 1
plt.show()


# plt.plot(loss_nosig, label='No signal')
# plt.plot(los_sig, label='Signal')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
# plt.show()

# # -----------------------------------------------------------------------
# # --------------------------    MODEL     -------------------------------
# # -----------------------------------------------------------------------
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')
plt.show()
# ================ summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
