
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd


# ============================================================
# =================creating DATA==============================
# ============================================================

def manPsignal(signal, bg, amp):
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

    def gaussian_generator(mu, sigma, amp, bground):
        result = 0.5*(amp*((stats.norm.pdf(indexv, mu, sigma)) + stats.norm.pdf(indexv+1, mu, sigma))) + bground
        return result

    def fluctuation_manufacture(numofevents_1, bground):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
        data_bg = np.random.poisson(1 * bground, np.array(numofevents_1).shape)

        return data_si_bg, data_bg

    # ============================================================
    # =================creating TRAINING data=====================
    # ============================================================

    bgconst = bg
    mu = 40
    amp1 = amp
    sigma_training = signal
    bground_training = np.zeros(precision) + bgconst
    numofpredictions = 60000
    normalization = 2 * math.sqrt(bgconst)

    gaussian_test = gaussian_generator(mu, sigma_training, amp1, bground_training)

    from1 = int(mu - 1.5*sigma_training)
    to = int(mu + 1.5*sigma_training)

    gausaray=[]
    for _ in range(numofpredictions):
        gausaray.append(gaussian_test)

    aa, bb = fluctuation_manufacture(gausaray, bgconst)
    aanormal = (np.array(aa) - bgconst) / normalization
    manual_01 = np.trapz(aanormal[:, ], axis=1)

    bbnormal = (np.array(bb) - bgconst) / normalization
    manual_00 = np.trapz(bbnormal[:, ])
    manualmedian_00 = np.median(manual_01)

    manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)

    print('\033[1m', 'manual calc', '\033[0m')
    print('#of values after median manual', manualsum_00)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))
    pvalue = manualsum_00 / numofpredictions
    # print('z-score', abs(st.norm.ppf(pvalue)), 'sigma')
    # return abs(st.norm.ppf(pvalue))
    return pvalue


# ======== p value VS background 3 different sigma===============

# bg1222 = []
# pvalue2 = []
#
#
# bg = 100
# for i in range(21):
#      bg1222.append(bg)
#      pvalue2.append(manPsignal(5, bg, 80))
#      bg = bg + 10
#
#
# bg1222 = np.array(bg1222)
# pvalue2 = np.array(pvalue2)
#
#
# plt.title('P-value VS Background')
# plt.ylabel('P-value')
# plt.yscale('log')
# plt.xlabel('Background [Events/GeV]')
# plt.plot(bg1222, pvalue2, label='\u03C3=5')
# plt.legend()
# plt.grid()
# plt.show()
#
# df1 = pd.DataFrame([bg1222, pvalue2],
#                    index=['bg', 'sigma5'])
# df1.to_excel("1peakNaeive.xlsx")


# ========================amp================================
bg1222 = []
pvalue1 = []
pvalue2 = []
pvalue3 = []

amp = 10
for i in range(13):

     bg1222.append(amp)
     pvalue2.append(manPsignal(5, 200, amp))
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

df1 = pd.DataFrame([bg1222, pvalue2 ],
                   index=['amp', 'sigma5'])
df1.to_excel("1peakNaeiveAMP.xlsx")