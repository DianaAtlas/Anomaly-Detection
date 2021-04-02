
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
from termcolor import colored

# ============================================================
# =================creating DATA==============================
# ============================================================


def manPvalue8exp(signal, amp1, bg):
    precision = 500
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)

    def gaussian_generator8(mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma, amp, exp):

        result = 0.5*(amp*((stats.norm.pdf(indexv, mu1, sigma)) + stats.norm.pdf(indexv+1, mu1, sigma) +
                           (stats.norm.pdf(indexv, mu2, sigma)) + stats.norm.pdf(indexv+1, mu2, sigma) +
                           (stats.norm.pdf(indexv, mu3, sigma)) + stats.norm.pdf(indexv+1, mu3, sigma) +
                           (stats.norm.pdf(indexv, mu4, sigma)) + stats.norm.pdf(indexv+1, mu4, sigma) +
                           (stats.norm.pdf(indexv, mu5, sigma)) + stats.norm.pdf(indexv+1, mu5, sigma) +
                           (stats.norm.pdf(indexv, mu6, sigma)) + stats.norm.pdf(indexv+1, mu6, sigma) +
                           (stats.norm.pdf(indexv, mu7, sigma)) + stats.norm.pdf(indexv+1, mu7, sigma) +
                           (stats.norm.pdf(indexv, mu8, sigma)) + stats.norm.pdf(indexv+1, mu8, sigma))) + exp

        return result

    def fluctuation_manufacture(numofevents_1, exp):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
        data_bg = np.random.poisson(1 * exp, np.array(numofevents_1).shape)

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


    mu2 = 30
    mu3 = 90
    Mu1 = 150
    mu4 = 210
    mu5 = 270
    mu6 = 330
    mu7 = 390
    mu8 = 450
    amp = amp1

    exp = bg * np.exp(-indexv / 90)
    normalization = 2 * np.sqrt(exp)
    exp = np.array(exp)

    sigma_training = signal
    numofpredictions = 60000

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    gaussian_test = gaussian_generator8(Mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma_training, amp, exp)


    expar = []
    normalizationar = []
    gausaray = []

    for _ in range(numofpredictions):
        gausaray.append(gaussian_test)
        expar.append(exp)
        normalizationar.append(normalization)

    aa, bb = fluctuation_manufacture(gausaray, exp)
    print('aa', np.array(aa).shape)
    print('exp', np.array(exp).shape)
    aanorm = (np.array(aa) - expar) / normalizationar
    bbnorm = (np.array(bb) - expar) / normalizationar


    manual_01 = (np.trapz(aanorm[:], axis=1))
    manual_00 = (np.trapz(bbnorm[:], axis=1))

    print('manual01', np.array(manual_01).shape, manual_01)
    print('manual00', np.array(manual_00).shape, manual_00)


    manualmedian_00 = np.median(manual_01)
    print('manualmedian_00', manualmedian_00)
    manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)
    print('manualsum_00', manualsum_00)

    print('\033[1m', 'manual calc', '\033[0m')
    print('#of values after median manual', manualsum_00)
    print('amp', amp)
    print('bg', bg)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))

    pvalue = manualsum_00 / numofpredictions
    # print('z-score', abs(st.norm.ppf(pvalue)), 'sigma')
    # return abs(st.norm.ppf(pvalue))

    return pvalue


# ======== p value VS background ===============

bg1222 = []
pvalue2 = []


bg = 180
for i in range(30):

     bg1222.append(bg)
     pvalue2.append(manPvalue8exp(5, 10, bg))
     bg = bg + 7

bg1222 = np.array(bg1222)
pvalue2 = np.array(pvalue2)

plt.title('EXPONENTIAL Naieve')
plt.ylabel('P-Value')
plt.yscale('log')
plt.xlabel('Background Magnitude [Events/GeV]')
plt.plot(bg1222, pvalue2, label='Naieve')
plt.legend()
plt.grid()
plt.show()


df1 = pd.DataFrame([bg1222, pvalue2],
                   index=['bg', 'sigma5'])
df1.to_excel("8expNaieve_Bg.xlsx")

# =======p value VS signals amp ================
# bg1222 = []
# pvalue2 = []
#
#
# amp = 1
# for i in range(30):
#
#      bg1222.append(amp)
#      pvalue2.append(manPvalue8exp(5, amp))
#      amp = amp + 1
#
# bg1222 = np.array(bg1222)
# pvalue2 = np.array(pvalue2)
#
# plt.title('EXPONENTIAL Naieve')
# plt.ylabel('P-Value')
# plt.yscale('log')
# plt.xlabel('Signal Magnitude [Events/GeV]')
# plt.plot(bg1222, pvalue2, label='Naieve')
# plt.legend()
# plt.grid()
# plt.show()
#
#
# df1 = pd.DataFrame([bg1222, pvalue2],
#                    index=['amp', 'sigma5'])
# df1.to_excel("8expNaieve_amp.xlsx")




