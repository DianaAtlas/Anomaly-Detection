
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import scipy.stats as stats
import math
import random
import matplotlib.pyplot as plt
import pandas as pd


def unman02(amp, bg):
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
    Amp = amp
    sigma_training = 5
    bground_training = np.zeros(precision) + bgconst
    numofpredictions = 60000

    normalization = 2 * math.sqrt(bgconst)

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    data_testing_01 = []

    manual_01 = []
    manual_00 = []

    for _ in range(numofpredictions):
        aa, bb = fluctuation_manufacture(sigma_training, Amp, bground_training)
        data_testing_01.append(aa)
        aanormal = (np.array(aa) - bgconst) / normalization
        manual_01.append(np.trapz(aanormal))
        bbnormal = (np.array(bb) - bgconst) / normalization
        manual_00.append(np.trapz(bbnormal))

    # ============================================================
    # ======================manual p value calc==================
    # ============================================================

    manualmedian_00 = np.median(manual_01)
    manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)

    print('\033[1m', 'manual calc', '\033[0m')
    print('#of values after median manual', manualsum_00)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))
    pvalue = manualsum_00 / numofpredictions

    return pvalue


# # =============================amp===========================
bg1222 = []
pvalue1 = []
pvalue2 = []
pvalue3 = []

amp = 10
for i in range(50):

     bg1222.append(amp)
     pvalue2.append(unman02(amp, 200))
     amp = amp + 10


bg1222 = np.array(bg1222)

pvalue2 = np.array(pvalue2)


plt.title('P-value VS Signal Magnitude')
plt.ylabel('P-value ')
plt.yscale('log')
plt.xlabel('Signal Magnitude [Events/GeV]')
plt.plot(bg1222, pvalue2, label='\u03C3=5')

plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue2],
                   index=['amp', 'sigma5'])
df1.to_excel("UN_AMP_MANUAL02.xlsx")

##################################################################
##########################     BG    #############################
##################################################################
#
# pvalue2 = []
# bg = 10
# loops = 10
# bg1222 = []
#
# for i in range(75):
#     bg1222.append(bg)
#     pvalue2.append(unman02(80, bg))
#     bg = bg + 5

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
# df1.to_excel("manual02_bg.xlsx")
