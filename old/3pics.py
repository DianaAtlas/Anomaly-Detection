


#==================================== 3 pics====================================


import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import scipy.stats as stats
import math
import warnings
from keras import optimizers
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import regularizers
from scipy.integrate import quad, simps
from keras.layers import LeakyReLU
# ============================================================
# =================creating DATA==============================
# ============================================================

# ========sample the statistic fluctuations======


def gaussian_generator3(mu1, mu2, mu3, sigma, amp, bground):
    numofevents_1 = []
    for i in range(precision):
        numofevents_1.append((0.5 *(amp * (stats.norm.pdf(i, mu1, sigma) + stats.norm.pdf(i + 1, mu1, sigma)))
                             +0.5 *(amp * (stats.norm.pdf(i, mu2, sigma) + stats.norm.pdf(i + 1, mu2, sigma)))
                             +0.5 *(amp *(stats.norm.pdf(i, mu3, sigma) + stats.norm.pdf(i + 1, mu3, sigma)) + 2 * bground)))

    return numofevents_1


def fluctuation_manufacture(numofevents_1, bground, precision):
    i = 0
    data_si_bg = []
    data_bg = []

    while i < precision:
        statistic_fluc = np.random.poisson(numofevents_1[i], 1)
        data_si_bg.append(statistic_fluc[0])
        statistic_fluc_bg = np.random.poisson(1 * bground, 1)
        data_bg.append(statistic_fluc_bg[0])
        i += 1

    return data_si_bg, data_bg



# ---------- sum all values above
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

precision = 150
amp_training = 60
const_training = 200

mu_training_1 = 10
mu_training_2 = 40
mu_training_3 = 70

variance_training = 5
sigma_training = math.sqrt(variance_training)
x = np.linspace(0, 150, 150)
bground_training = const_training + 0 * x
si_bg = bground_training + stats.norm.pdf(x, mu_training_1, sigma_training) + stats.norm.pdf(x, mu_training_2,sigma_training) + stats.norm.pdf(x, mu_training_3,sigma_training)


gaussian_training = gaussian_generator3(mu_training_1,mu_training_2, mu_training_3, sigma_training, amp_training, bground_training)
data_combined = []
label_combined = []

for _ in range(2000):
    a, b = fluctuation_manufacture(gaussian_training, bground_training, precision)
    data_combined.append(a)
    label_combined.append(1)
    data_combined.append(b)
    label_combined.append(0)

print( data_combined)
data_combined_norm = np.array(data_combined) / 1000    # normalizing the data


# ============================================================
# =================creating TEST data=========================
# ============================================================
numofexampls = 10000

mu_test_1 = 10
mu_test_2 = 40
mu_test_3 = 70

amp_testing = 60
const_test = 200
variance_test = 9
epoch = 400

sigma_test = math.sqrt(variance_training)
x_test = np.linspace(0, 150, 150)
bground_test = const_training + 0 * x

data_testing_01 = []
data_testing_00 = []
label_testing = []
gaussian_test = gaussian_generator3(mu_test_1, mu_test_2, mu_test_3, sigma_test, amp_testing, bground_test)


from1 = int(mu_test_1 - 3*sigma_test)
to1 = int(mu_test_1 + 3*sigma_test)

from2 = int(mu_test_2 - 3*sigma_test)
to2 = int(mu_test_2 + 3*sigma_test)

from3 = int(mu_test_3 - 3*sigma_test)
to3 = int(mu_test_3 + 3*sigma_test)


manual_01 = []
manual_00 = []

for _ in range(numofexampls):
    aa, bb = fluctuation_manufacture(gaussian_test, bground_test, precision)
    data_testing_01.append(aa)
    label_testing.append(1)
    aanormal = np.array(aa) / 1000
    manual_01.append(np.trapz(aanormal[from1:to1])+np.trapz(aanormal[from2:to2])+np.trapz(aanormal[from3:to3]))


for _ in range(numofexampls):
    aa, bb = fluctuation_manufacture(gaussian_test, bground_test, precision)
    data_testing_00.append(bb)
    label_testing.append(0)
    bbnormal = np.array(bb) / 1000
    manual_00.append(np.trapz(bbnormal[from1:to1])+np.trapz(bbnormal[from2:to2])+np.trapz(bbnormal[from3:to3]))

plt.title('Histogram of manual calc')
x00, t0 = np.histogram(manual_00, bins=50)
x01, t1 = np.histogram(manual_01, bins=50)
plt.plot(t0[0:-1], x00)
plt.plot(t1[0:-1], x01)
plt.show

plt.plot(aa[:0])
plt.ylabel("<dN/dM>")
plt.xlabel('M')
plt.show()

data_testing = data_testing_00 + data_testing_01
data_testing_01_norm = np.array(data_testing_01) / 1000
data_testing_00_norm = np.array(data_testing_00) / 1000
data_testing_norm = np.array(data_testing) / 1000

# ============================================
# ===================from here is NN==========
# ============================================

model = Sequential()
model.add(Dense(10, batch_size=50,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))    # Kernel Regularization regulates the weights


model.add(Dense(5, batch_size=50,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))


model.add(Dense(1, batch_size=50,
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(0.001)))

optimizer = optimizers.rmsprop(lr=0.0001)
model.compile(loss='binary_crossentropy',
              optimizer = optimizer,
              metrics=['accuracy'])

history = model.fit(np.array(data_combined_norm),
                    np.array(label_combined),
                    epochs=epoch,
                    validation_split=0.2,
                    verbose=1)
print(model.summary())

print('NN predicted', model.predict([np.array(data_testing_norm)]))
print((label_combined))
plt.plot(model.predict([np.array(data_testing_norm)]))
plt.title('model prediction')
plt.show()
print(model.predict(np.array(data_testing_00_norm)))
plt.title('model prediction, should predict everthing without signal')
plt.plot(model.predict(np.array(data_testing_00_norm)))
plt.show()

# ===================Getting the weights============
W_0 = model.layers[0].get_weights()[0]
W_1 = model.layers[1].get_weights()[0]

f, (ax1, ax2) = plt.subplots(2, 1)
plt.xlabel('M')
plt.ylabel('weight')
ax1.set_title('Weights Layer 0')
ax1.plot(W_0)
plt.xlabel('M')
plt.ylabel('weight')
ax2.set_title('Weights Layer 1')
ax2.plot(W_1)
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
t=1

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


#===================================================
#-----------calculating the p value via ml----------
#===================================================

median_00 = np.median(model.predict(data_testing_01_norm))

sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00_norm)), median_00)

print ('\033[1m', 'ml calc', '\033[0m')
print('#of values after median', sum_00)
print('presantage', sum_00/numofexampls)

#============================================================
# ======================manual p value calc==================
#============================================================
manualmedian_00 = np.median(manual_01)

manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)


print('\033[1m', 'manual calc', '\033[0m')
print('#of values after median manual', manualsum_00)
print('presantage', manualsum_00 / numofexampls)

# ------------calculating the integral value via trapz integral
#
# median_01 = np.median(model.predict(data_testing_00_norm))
# median_00 = np.median(model.predict(data_testing_01_norm))
#
# num_of_bins = 1000
#
# y00, x00 = np.histogram(model.predict(np.array(data_testing_00_norm)), bins=num_of_bins)
# y01, x01 = np.histogram(model.predict(np.array(data_testing_01_norm)), bins=num_of_bins)
#
# binindex01 = int(median_01 * num_of_bins)
# binindex00 = int(median_00 * num_of_bins)
#
# print('Integrating via Traps:')
# print(np.trapz(y00[binindex00:], dx=1.0/num_of_bins))
# print(np.trapz(y01[0:binindex01], dx=1.0/num_of_bins))
#
# print('Integrating via Simps:')
# print(simps(y00[binindex00:], dx=1.0/num_of_bins))
# print(simps(y01[0:binindex01], dx=1.0/num_of_bins))