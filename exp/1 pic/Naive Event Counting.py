
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd


# ============================================================
# =================creating DATA==============================
# ============================================================

def NaiveExp(mu):
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


    def gaussian_generator(mu, sigma, amp, bground):
        result = 0.5 * (amp * ((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv + 1, mu, sigma))) + bground
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

    mu = mu
    amp1 = 150
    sigma_training = 5

    numofpredictions = 60000
    exp = 200 * np.exp(-indexv / 90)
    normalization = 2 * np.sqrt(exp)
    exp = np.array(exp)
    gaussian_test = gaussian_generator(mu, sigma_training, amp1, exp)

    expar = []
    normalizationar = []
    gausaray = []

    for _ in range(numofpredictions):
        gausaray.append(gaussian_test)
        expar.append(exp)
        normalizationar.append(normalization)

    aa, bb = fluctuation_manufacture(gausaray, exp)
    aanormal = (np.array(aa) - expar) / normalizationar
    manual_01 = np.trapz(aanormal[:, ], axis=1)

    bbnormal = (np.array(bb) - expar) / normalizationar
    manual_00 = np.trapz(bbnormal[:, ])
    manualmedian_00 = np.median(manual_01)

    manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)

    print('\033[1m', 'manual calc', '\033[0m')
    print('#of values after median manual', manualsum_00)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00) / numofpredictions)
    pvalue = manualsum_00 / numofpredictions

    return pvalue



# ========================mu================================
bg1222 = []
pvalue = []


mass = 10
for i in range(20):

     bg1222.append(mass)
     pvalue.append(NaiveExp(mass))
     mass = mass + 5


bg1222 = np.array(bg1222)
pvalue = np.array(pvalue)



plt.title('P-value VS Signal Magnitude')
plt.ylabel('P-value ')
plt.yscale('log')
plt.xlabel('Signal Magnitude [Events/GeV]')
plt.plot(bg1222, pvalue, label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()

df1 = pd.DataFrame([bg1222, pvalue],
                   index=['mass', 'sigma5'])
df1.to_excel("NaiveMass.xlsx")