

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
import random


# ============================================================
# =================creating DATA==============================
# ============================================================

precision = 100
indexv = []
for i in range(precision):
    indexv.append(i)

indexv = np.array(indexv)



def fluctuation_manufacture( sigma, amp, bground):
    mu = random.uniform(10,90)
    signalshape = 0.5 * (amp * ((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv + 1, mu, sigma))) + bground
    np.array(signalshape)
    data_si_bg = np.random.poisson(signalshape, np.array(signalshape).size)
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
Mu = 40
Amp = 100
variance_training = 5
sigma_training = math.sqrt(variance_training)
x = np.linspace(Mu - 10 * sigma_training, Mu + 10 * sigma_training, 100)
bground_training = np.zeros(precision) + bgconst
numoftraning = 10000
numofpredictions = 60000
epoch = 150
normalization = 2 * math.sqrt(bgconst)

si_bg = bground_training + stats.norm.pdf(x, Mu, sigma_training)


data_combined = []
label_combined = []

for _ in range(numoftraning):
    a, b = fluctuation_manufacture(sigma_training, Amp, bground_training)
    data_combined.append(a)
    label_combined.append(1)
    data_combined.append(b)
    label_combined.append(0)

data_combined_norm = (np.array(data_combined) - bgconst) / normalization

plt.title('Signal with fluctuations')
plt.plot(a)
plt.legend()
plt.xlabel('M[GeV]')
plt.ylabel('Events')
plt.grid()
plt.show()

# ============================================================
# =================creating TEST data=========================
# ============================================================


data_testing_01 = []
data_testing_00 = []
label_testing = []


from1 = int(Mu - 1.5*sigma_training)
to = int(Mu + 1.5*sigma_training)

manual_01 = []
manual_00 = []


for _ in range(numofpredictions):
    aa, bb = fluctuation_manufacture(sigma_training, Amp, bground_training)
    data_testing_01.append(aa)
    label_testing.append(1)
    aanormal = (np.array(aa) - bgconst) / normalization
    manual_01.append(np.trapz(aanormal))

for _ in range(numofpredictions):
    aa, bb = fluctuation_manufacture(sigma_training, Amp, bground_training)
    data_testing_00.append(bb)
    label_testing.append(0)
    bbnormal = (np.array(bb) - bgconst) / normalization
    manual_00.append(np.trapz(bbnormal))


print(np.array(aa).shape, aa)
# plt.plot(aa)
# plt.ylabel("<dN/dM>")
# plt.xlabel('M')
# # plt.show()

plt.title('Histogram of manual calc')
x00, t0 = np.histogram(manual_00, bins=100)
x01, t1 = np.histogram(manual_01, bins=100)
plt.plot(t0[0:-1], x00)
plt.plot(t1[0:-1], x01)
plt.show


data_testing = data_testing_00 + data_testing_01
data_testing_01_norm = (np.array(data_testing_01) - bgconst) / normalization
data_testing_00_norm = (np.array(data_testing_00) - bgconst) / normalization
data_testing_norm = (np.array(data_testing) - bgconst) / normalization



#-----------calculating the p value via ml----------

median_00 = np.median(model.predict(data_testing_01_norm))
sum_00 = sum_all_values_over(model.predict(np.array(data_testing_00_norm)), median_00)

print ('\033[1m', 'ml calc', '\033[0m')
print('#of values after median', sum_00)
print('presantage', sum_00 / numofpredictions, '+-', math.sqrt(sum_00 / numofpredictions))


#============================================================
# ======================manual p value calc==================
#============================================================

manualmedian_00 = np.median(manual_01)
manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)

print('\033[1m', 'manual calc', '\033[0m')
print('#of values after median manual', manualsum_00)
print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))




