

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


# ============================================================
# =================creating DATA==============================
# ============================================================

precision = 500
indexv = []
for i in range(precision):
    indexv.append(i)

indexv = np.array(indexv)


def gaussian_generator8(mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma, amp, bground):
    result = []
    result = 0.5*(amp*((stats.norm.pdf(indexv, mu1, sigma)) + stats.norm.pdf(indexv+1, mu1, sigma) +
                       (stats.norm.pdf(indexv, mu2, sigma)) + stats.norm.pdf(indexv+1, mu2, sigma) +
                       (stats.norm.pdf(indexv, mu3, sigma)) + stats.norm.pdf(indexv+1, mu3, sigma) +
                       (stats.norm.pdf(indexv, mu4, sigma)) + stats.norm.pdf(indexv+1, mu4, sigma) +
                       (stats.norm.pdf(indexv, mu5, sigma)) + stats.norm.pdf(indexv+1, mu5, sigma) +
                       (stats.norm.pdf(indexv, mu6, sigma)) + stats.norm.pdf(indexv+1, mu6, sigma) +
                       (stats.norm.pdf(indexv, mu7, sigma)) + stats.norm.pdf(indexv+1, mu7, sigma) +
                       (stats.norm.pdf(indexv, mu8, sigma)) + stats.norm.pdf(indexv+1, mu8, sigma))) + bground

    return result


def fluctuation_manufacture(numofevents_1, bground,):
    np.array(numofevents_1)
    data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).size)
    data_bg = np.random.poisson(1 * bground, np.array(bground).size)

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

bgconst = 100

mu2 = 30
mu3 = 90
Mu1 = 150
mu4 = 210
mu5 = 270
mu6 = 330
mu7 = 390
mu8 = 450

Amp = 30
sigma_training = 5

bground_training = np.zeros(precision) + bgconst
numoftraning = 10000
numofpredictions = 60000
epoch = 100
normalization = 2 * math.sqrt(bgconst)
signalshape = gaussian_generator8(Mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma_training, Amp, bground_training)


plt.title('8 Signals')
plt.ylabel('Events / GeV')
plt.xlabel('m [GeV]')
plt.grid()
plt.plot(signalshape)
plt.show()

data_combined = []
label_combined = []

for _ in range(numoftraning):
    a, b = fluctuation_manufacture(signalshape, bground_training)
    data_combined.append(a)
    label_combined.append(1)
    data_combined.append(b)
    label_combined.append(0)

data_combined_norm = (np.array(data_combined) - bgconst) / normalization

plt.title('Signal and Fluctuations')
plt.plot(a)
plt.legend()
plt.grid()
plt.xlabel('M [GeV]')
plt.ylabel('# of Events')
plt.show()

# ============================================================
# =================creating TEST data=========================
# ============================================================


data_testing_01 = []
data_testing_00 = []
label_testing = []
gaussian_test = gaussian_generator8(Mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma_training, Amp, bground_training)


from1 = int(Mu1 - 1.5*sigma_training)
to = int(Mu1 + 1.5*sigma_training)

from2 = int(mu2 - 1.5*sigma_training)
to2 = int(mu2 + 1.5*sigma_training)

from3 = int(mu3 - 1.5*sigma_training)
to3 = int(mu3 + 1.5*sigma_training)

from4 = int(mu4 - 1.5*sigma_training)
to4 = int(mu4 + 1.5*sigma_training)

from5 = int(mu5 - 1.5*sigma_training)
to5 = int(mu5 + 1.5*sigma_training)

from6 = int(mu6 - 1.5*sigma_training)
to6 = int(mu6 + 1.5*sigma_training)

from7 = int(mu7 - 1.5*sigma_training)
to7 = int(mu7 + 1.5*sigma_training)

from8 = int(mu8 - 1.5*sigma_training)
to8 = int(mu8 + 1.5*sigma_training)


manual_01 = []
manual_00 = []


for _ in range(numofpredictions):
    aa, bb = fluctuation_manufacture(gaussian_test, bground_training)
    data_testing_01.append(aa)
    label_testing.append(1)
    aanormal = (np.array(aa) - bgconst) / normalization
    manual_01.append(sum(aanormal[from1:to]) + sum(aanormal[from2:to2])
                     + sum(aanormal[from3:to3]) + sum(aanormal[from4:to4])
                     + sum(aanormal[from5:to5]) + sum(aanormal[from6:to6])
                     + sum(aanormal[from7:to7]) + sum(aanormal[from8:to8]))

for _ in range(numofpredictions):
    aa, bb = fluctuation_manufacture(gaussian_test, bground_training)
    data_testing_00.append(bb)
    label_testing.append(0)
    bbnormal = (np.array(bb) - bgconst) / normalization
    manual_00.append(sum(bbnormal[from1:to]) + sum(bbnormal[from2:to2])
                     + sum(bbnormal[from3:to3]) + sum(bbnormal[from4:to4])
                     + sum(bbnormal[from5:to5]) + sum(bbnormal[from6:to6])
                     + sum(bbnormal[from7:to7]) + sum(bbnormal[from8:to8]))

print(np.array(aa).shape, aa)
plt.plot(aa)
plt.ylabel("<dN/dM>")
plt.xlabel('M')
plt.show()

plt.title('Histogram of manual calc')
x00, t0 = np.histogram(manual_00, bins=100)
x01, t1 = np.histogram(manual_01, bins=100)
plt.plot(t0[0:-1], x00)
plt.plot(t1[0:-1], x01)
plt.show()


data_testing = data_testing_00 + data_testing_01
data_testing_01_norm = (np.array(data_testing_01) - bgconst) / normalization
data_testing_00_norm = (np.array(data_testing_00) - bgconst) / normalization
data_testing_norm = (np.array(data_testing) - bgconst) / normalization

plt.plot(data_testing_01[0], label='fluctuations')
plt.plot(signalshape, label ='original signal')
plt.title('Signal with fluctuations')
plt.grid()
plt.xlabel('M[GeV]')
plt.ylabel('Events')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
plt.show()
#-----------------------------------------------------------------------
#------ 2 model layer---------------------------------------------------
#-----------------------------------------------------------------------

model = Sequential()

model.add(Dense(1, batch_size=5000,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))    # Kernel Regularization regulates the weights

model.add(Dense(1, activation='sigmoid',
                kernel_regularizer=regularizers.l2(0.001)))

rmspop = optimizers.rmsprop(lr=0.001, rho=0.9)

model.compile(loss='binary_crossentropy',
              optimizer=rmspop,
              metrics=['accuracy'])

history = model.fit(np.array(data_combined_norm),
                    np.array(label_combined),
                    epochs=epoch,
                    validation_split=0.2,
                    verbose=1,
                    shuffle=True)
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
plt.xlabel('M')
plt.ylabel('weight')
plt.show()

# ====================Histogram=============
plt.xlabel('value', color='#1C2833')
plt.ylabel('# of times counted', color='#1C2833')

plt.title('Histogram')

y, bi = np.histogram(model.predict(np.array(data_testing_01_norm)), bins=np.arange(50+1)/50)
y2, bi2 = np.histogram(model.predict(np.array(data_testing_00_norm)), bins=np.arange(50+1)/50)

plt.plot(bi[0:-1], y)
plt.plot(bi2[0:-1], y2)

plt.show()

difference = 1-y
t = 1
plt.show()

# =============== summarize history for accuracy

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


#-----------calculating the p value via ml----------

median_00 = np.median(model.predict(data_testing_01_norm))
sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00_norm)), median_00)

print ('\033[1m', 'ml calc', '\033[0m')
print('#of values after median', sum_00)
print('presantage', sum_00 / numofpredictions, '+-', math.sqrt(sum_00 / numofpredictions))

# median_01 = np.median(model.predict(data_testing_00_norm))
# sum_01 = sum_all_values_under(model.predict(np.array(data_testing_01_norm)), median_01)
# print(sum_01)
#============================================================
# ======================manual p value calc==================
#============================================================

manualmedian_00 = np.median(manual_01)
manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)

print('\033[1m', 'manual calc', '\033[0m')
print('#of values after median manual', manualsum_00)
print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))




