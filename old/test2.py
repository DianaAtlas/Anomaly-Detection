
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
def pvalueunknownloca(sigmaa,bg):
    precision = 100
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)


    def fluctuation_manufacture( sigma, amp, bground):
        mu = random.uniform(10, 90)
        signalshape = 0.5 * (amp * ((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv + 1, mu, sigma))) + bground
        np.array(signalshape)
        data_si_bg = np.random.poisson(signalshape, np.array(signalshape).size)
        data_bg = np.random.poisson(1 * bground, np.array(bground).size)

        return data_si_bg, data_bg



    def sum_all_values_over_manual(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i] >= number:
                value_returned += 1

        return value_returned
    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================


    bgconst = bg
    Amp = 100
    numofpredictions = 60000
    bground_training = np.zeros(precision) + bgconst
    normalization = 2 * math.sqrt(bgconst)
    sigma_training = sigmaa


    data_testing_01 = []
    data_testing_00 = []

    manual_01 = []
    manual_00 = []

    for _ in range(numofpredictions):
        aa, bb = fluctuation_manufacture(sigma_training, Amp, bground_training)
        data_testing_01.append(aa)
        data_testing_00.append(bb)
        aanormal = (np.array(aa) - bgconst) / normalization
        manual_01.append(np.trapz(aanormal))
        bbnormal = (np.array(bb) - bgconst) / normalization
        manual_00.append(np.trapz(bbnormal))


    #============================================================
    # ======================manual p value calc==================
    #============================================================

    manualmedian_00 = np.median(manual_01)
    manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)

    print('\033[1m', 'manual calc', '\033[0m')
    print('#of values after median manual', manualsum_00)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))

    return manualsum_00 / numofpredictions



#======== p value VS sigma 3 different bg===============
sig1222 = []
pvalue1 = []
pvalue2 = []
pvalue3 = []
sigma1 = 1


for i in range(100):

     sigma1 = sigma1 + 0.5
     sig1222.append(sigma1)
     pvalue1.append(pvalueunknownloca(sigma1, 150))
     pvalue2.append(pvalueunknownloca(sigma1, 50))
     pvalue3.append(pvalueunknownloca(sigma1, 200))


bg1222 = np.array(sig1222)
pvalue1 = np.array(pvalue1)
pvalue2 = np.array(pvalue2)
pvalue3 = np.array(pvalue3)

print(pvalue1)
print(sig1222)

plt.title('P-value VS Signal Width')
plt.ylabel('P-value')
plt.xlabel('Signal Width ')
plt.plot(sig1222, pvalue1, label='Amp=150')
plt.plot(sig1222, pvalue2, label='Amp=50')
plt.plot(sig1222, pvalue3, label='Amp=200')
plt.legend()
plt.show()