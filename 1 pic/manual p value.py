
import numpy as np
import scipy.stats as stats
import math
import warnings
import random
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt

# ============================================================
# =================creating DATA==============================
# ============================================================
def manPsignal(signal,bg):
    precision = 100
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)

    x = np.linspace(0, precision - 1, num=precision)

    def sum_all_values_over_manual(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i] >= number:
                value_returned += 1

        return value_returned

    def gaussian_generator(mu, sigma, amp, bground):
        result = 0.5*(amp*((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv+1, mu, sigma))) + bground
        return result


    def fluctuation_manufacture(numofevents_1, bground,):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).size)
        data_bg = np.random.poisson(1 * bground, np.array(bground).size)

        return data_si_bg, data_bg


    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================

    bgconst = bg
    Mu = 40
    Amp = 80
    sigma_training = signal
    x = np.linspace(Mu - 10 * sigma_training, Mu + 10 * sigma_training, 100)
    bground_training = np.zeros(precision) + bgconst
    numofpredictions = 40000
    normalization = 2 * math.sqrt(bgconst)

    data_testing_01 = []
    data_testing_00 = []
    label_testing = []
    gaussian_test = gaussian_generator(Mu, sigma_training, Amp, bground_training)


    from1 = int(Mu - 1.5*sigma_training)
    to = int(Mu + 1.5*sigma_training)

    manual_01 = []
    manual_00 = []

    for _ in range(numofpredictions):
        aa, bb = fluctuation_manufacture(gaussian_test, bground_training)
        data_testing_01.append(aa)
        label_testing.append(1)
        aanormal = (np.array(aa) - bgconst) / normalization
        manual_01.append(np.trapz(aanormal[from1:to]))

    for _ in range(numofpredictions):
        aa, bb = fluctuation_manufacture(gaussian_test, bground_training)
        data_testing_00.append(bb)
        label_testing.append(0)
        bbnormal = (np.array(bb) - bgconst) / normalization
        manual_00.append(np.trapz(bbnormal[from1:to]))

    #============================================================
    # ======================manual p value calc==================
    #============================================================

    manualmedian_00 = np.median(manual_01)
    manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)

    print('\033[1m', 'manual calc', '\033[0m')
    print('#of values after median manual', manualsum_00)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))

    return(manualsum_00 / numofpredictions)

#======== p value VS background 3 different sigma===============

bg1222 = []
pvalue1=[]
pvalue2=[]
pvalue3=[]


bg = 75
for i in range(80):

     bg = bg + 5
     bg1222.append(bg)
     pvalue1.append(manPsignal(1, bg))
     pvalue2.append(manPsignal(5, bg))
     pvalue3.append(manPsignal(10, bg))

bg1222 = np.array(bg1222)
pvalue1 = np.array(pvalue1)
pvalue2 = np.array(pvalue2)
pvalue3 = np.array(pvalue3)


plt.title('P-value VS Background')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Background')
plt.plot(bg1222, pvalue2, label='sigma=5')
plt.plot(bg1222, pvalue3, label='sigma=10')
plt.plot(bg1222, pvalue1, label='sigma=1')
plt.legend()
plt.show()







#======== p value VS sigma 3 different bg===============
# sig1222 = []
# pvalue1 = []
# pvalue2 = []
# pvalue3 = []
# sigma1 = 1
#
#
# for i in range(50):
#
#      sigma1 = sigma1 + 0.5
#      sig1222.append(sigma1)
#      pvalue1.append(manPsignal(sigma1, 150))
#      pvalue2.append(manPsignal(sigma1, 50))
#      pvalue3.append(manPsignal(sigma1, 200))
#
#
# bg1222 = np.array(sig1222)
# pvalue1 = np.array(pvalue1)
# pvalue2 = np.array(pvalue2)
# pvalue3 = np.array(pvalue3)
#
# print(pvalue1)
# print(sig1222)
#
# plt.title('P-value VS Signal Width')
# plt.ylabel('P-value')
#plt.yscale('log')
# plt.xlabel('Signal Width ')
# plt.plot(sig1222, pvalue1, label='Amp=150')
# plt.plot(sig1222, pvalue2, label='Amp=50')
# plt.plot(sig1222, pvalue3, label='Amp=200')
# plt.legend()
# plt.show()