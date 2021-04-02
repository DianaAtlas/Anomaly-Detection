
import numpy as np
import scipy.stats as stats
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
# ============================================================
# =================creating DATA==============================
# ============================================================


def unpv(sigma, bg, amp):
    precision = 100
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)


    def fluctuation_manufacture(numofevents_1, bground):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
        data_bg = np.random.poisson(1 * bground, np.array(numofevents_1).shape)

        return data_si_bg, data_bg


    def gaussian_generator(mu, sigma, amp):
        result = 0.5 * (amp * ((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv + 1, mu, sigma))) + exp
        return result

    def sum_all_values_over_manual(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i] >= number:
                value_returned += 1

        return value_returned

    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================


    Amp = amp
    sigma_training = sigma
    exp = 200 * np.exp(-indexv / 50)
    normalization = 2 * np.sqrt(exp)
    exp = np.array(exp)
    numofpredictions = 60000

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    gausaray = []
    normalizationar = []
    expar = []

    for _ in range(numofpredictions):
       x = gaussian_generator(random.uniform(10, 90), sigma_training, Amp)
       gausaray.append(x)
       expar.append(exp)
       normalizationar.append(normalization)

    aa, bb = fluctuation_manufacture(gausaray, expar)
    aanormal = (np.array(aa) - expar) / normalizationar
    bbnormal = (np.array(bb) - expar) / normalizationar

    manfull_01 = []
    manfull_00 = []

    for i in range(60000):

        manfull_01.append(np.trapz(aanormal[i]))
        manfull_00.append(np.trapz(bbnormal[i]))

    manualmedian_00 = np.median(manfull_01)
    manualsum_00 = sum_all_values_over_manual(manfull_00, manualmedian_00)
    print('\033[1m', 'manual calc', '\033[0m')
    print('#of values after median manual', manualsum_00)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))
    pvalue = manualsum_00 / numofpredictions

    return pvalue

# ======== p value VS background ===============

bg1222 = []
pvalue2 = []

bg = 75
for i in range(45):
     bg1222.append()
     pvalue2.append(unpv(5, bg, 80))
     bg = bg + 4


bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)



plt.title('P-value VS Background')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Background [Events/GeV]')
plt.plot(bg1222, pvalue2, label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue2],
                   index=['bg', 'sigma5'])
df1.to_excel("UN_BG_MANUAL_full.xlsx")


# # =============================amp===========================
# bg1222 = []
# pvalue2 = []
#
#
# amp = 1
# for i in range(70):
#
#      bg1222.append(amp)
#      pvalue2.append(unpv(5, 200, amp))
#      amp = amp + 5
#
#
# bg1222 = np.array(bg1222)
# pvalue2 = np.array(pvalue2)
# print('bg1222', bg1222)
# print('pvalue2', pvalue2)
#
# plt.title('EXPONENT UNKNOWN LOCATION')
# plt.ylabel('P-value ')
# plt.yscale('log')
# plt.xlabel('Signal Magnitude [Events/GeV]')
# plt.plot(bg1222, pvalue2, label='Naieve MANUAL')
#
# plt.legend()
# plt.grid()
# plt.show()
#
# df1 = pd.DataFrame([bg1222, pvalue2],
#                    index=['amp','PVALUE'])
# df1.to_excel("UN_AMP_NAEIVE_exp.xlsx")