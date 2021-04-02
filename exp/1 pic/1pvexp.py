
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd

# ============================================================
# =================creating DATA==============================
# ============================================================

def manexp(signal, mu, amp):
    precision = 100
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)

    def sum_all_values_over_manual(array, number):
        value_returned = 0
        for i in range(len(array)):
            if array[i] >= number:
                value_returned += 1

        return value_returned

    def gaussian_generator(mu, sigma, amp, exp):
        result = 0.5*(amp*((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv+1, mu, sigma))) + exp
        return result

    def fluctuation_manufacture(numofevents_1, exp):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
        data_bg = np.random.poisson(1 * exp, np.array(numofevents_1).shape)

        return data_si_bg, data_bg

    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================

    mu = mu
    amp1 = 150
    sigma_training = signal

    numofpredictions = 60000
    exp = 200 * np.exp(-indexv / 90)
    normalization = 2 * np.sqrt(exp)
    exp = np.array(exp)
    gaussian_test = gaussian_generator(mu, sigma_training, amp1, exp)

    from1 = int(mu - 1.5*sigma_training)
    to = int(mu + 1.5*sigma_training)

    expar = []
    normalizationar = []
    gausaray = []

    for _ in range(numofpredictions):
        gausaray.append(gaussian_test)
        expar.append(exp)
        normalizationar.append(normalization)

    aa, bb = fluctuation_manufacture(gausaray, exp)
    aanormal = (np.array(aa) - expar) / normalizationar
    manual_01 = np.trapz(aanormal[:, from1:to], axis=1)

    bbnormal = (np.array(bb) - expar) / normalizationar
    manual_00 = np.trapz(bbnormal[:, from1:to])
    manualmedian_00 = np.median(manual_01)

    manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)

    print('\033[1m', 'manual calc', '\033[0m')
    print('#of values after median manual', manualsum_00)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))
    pvalue = manualsum_00 / numofpredictions
    # print('z-score', abs(st.norm.ppf(pvalue)), 'sigma')
    # return abs(st.norm.ppf(pvalue))
    return pvalue


# ======== p value VS  MU ==============

bg1222 = []
pvalue1 = []

mu = 10
for i in range(45):
     bg1222.append(mu)
     pvalue1.append(manexp(5, mu, 200))
     mu = mu + 2

bg1222 = np.array(bg1222)
pvalue1 = np.array(pvalue1)

plt.title('P-value VS Pic Location')
plt.ylabel('P-value')
plt.yscale('log')
plt.xlabel('Mass [Events/GeV]')
plt.plot(bg1222, pvalue1, label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue1],
                   index=['mass', 'sigma5'])
df1.to_excel("MUexpEVENT COUNTING.xlsx")


# ========================amp================================
# bg1222 = []
# pvalue1 = []
# pvalue2 = []
# pvalue3 = []
#
# amp = 10
# for i in range(80):
#
#      bg1222.append(amp)
#      pvalue2.append(manPsignal(5, 200, amp))
#      pvalue3.append(manPsignal(10, 200, amp))
#      pvalue1.append(manPsignal(3, 200, amp))
#      amp = amp + 5
#
#
# bg1222 = np.array(bg1222)
# pvalue1 = np.array(pvalue1)
# pvalue2 = np.array(pvalue2)
# pvalue3 = np.array(pvalue3)
#
#
# plt.title('P-value VS Signal Magnitude')
# plt.ylabel('P-value ')
# plt.yscale('log')
# plt.xlabel('Signal Magnitude [Events/GeV]')
# plt.plot(bg1222, pvalue2, label='\u03C3=5')
# plt.plot(bg1222, pvalue3, label='\u03C3=10')
# plt.plot(bg1222, pvalue1, label='\u03C3=3')
# plt.legend()
# plt.grid()
# plt.show()
#
# df1 = pd.DataFrame([bg1222, pvalue1, pvalue2, pvalue3],
#                    index=['amp', 'sigma3', 'sigma5', 'sigma10'])
# df1.to_excel("1picman.xlsx")