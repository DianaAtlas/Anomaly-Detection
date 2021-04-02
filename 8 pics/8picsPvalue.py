
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import scipy.stats as st
import pandas as pd

# ============================================================
# =================creating DATA==============================
# ============================================================


def manPvalue8(signal, baground, amp1):
    precision = 500
    indexv = []
    for i in range(precision):
        indexv.append(i)

    indexv = np.array(indexv)

    def gaussian_generator8(mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma, amp, bground):

        result = 0.5*(amp*((stats.norm.pdf(indexv, mu1, sigma)) + stats.norm.pdf(indexv+1, mu1, sigma) +
                           (stats.norm.pdf(indexv, mu2, sigma)) + stats.norm.pdf(indexv+1, mu2, sigma) +
                           (stats.norm.pdf(indexv, mu3, sigma)) + stats.norm.pdf(indexv+1, mu3, sigma) +
                           (stats.norm.pdf(indexv, mu4, sigma)) + stats.norm.pdf(indexv+1, mu4, sigma) +
                           (stats.norm.pdf(indexv, mu5, sigma)) + stats.norm.pdf(indexv+1, mu5, sigma) +
                           (stats.norm.pdf(indexv, mu6, sigma)) + stats.norm.pdf(indexv+1, mu6, sigma) +
                           (stats.norm.pdf(indexv, mu7, sigma)) + stats.norm.pdf(indexv+1, mu7, sigma) +
                           (stats.norm.pdf(indexv, mu8, sigma)) + stats.norm.pdf(indexv+1, mu8, sigma))) + bground

        return result

    def fluctuation_manufacture(numofevents_1, bground):
        np.array(numofevents_1)
        data_si_bg = np.random.poisson(numofevents_1, np.array(numofevents_1).shape)
        data_bg = np.random.poisson(1 * bground, np.array(numofevents_1).shape)

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

    baground = baground
    mu2 = 30
    mu3 = 90
    Mu1 = 150
    mu4 = 210
    mu5 = 270
    mu6 = 330
    mu7 = 390
    mu8 = 450
    amp = amp1
    sigma_training = signal
    bground_training = np.zeros(precision) + baground
    numofpredictions = 60000
    normalization = 2 * math.sqrt(baground)

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    gaussian_test = gaussian_generator8(Mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, sigma_training, amp, bground_training)

    from1 = int(Mu1 - 1.5*sigma_training)
    to = int(Mu1 + 1.5*sigma_training)

    from2 = int(mu2 - 1.5*sigma_training)
    to2 = int(mu2 + 1.5*sigma_training)

    from3 = int(mu3 - 1.5*sigma_training)
    to3 = int(mu3 + 1.5*sigma_training)

    from4 = int(mu4 - 1.5*sigma_training)
    to4 = int(mu4 + 1.5*sigma_training)

    from5 = int(mu5 - 1.5*sigma_training)
    to5 = int(mu5 + 1.5*sigma_training)

    from6 = int(mu6 - 1.5*sigma_training)
    to6 = int(mu6 + 1.5*sigma_training)

    from7 = int(mu7 - 1.5*sigma_training)
    to7 = int(mu7 + 1.5*sigma_training)

    from8 = int(mu8 - 1.5*sigma_training)
    to8 = int(mu8 + 1.5*sigma_training)

    gausaray = []
    for _ in range(numofpredictions):
        gausaray.append(gaussian_test)

    aa, bb = fluctuation_manufacture(gausaray, baground)
    aanorm = (np.array(aa) - baground) / normalization
    bbnorm = (np.array(bb) - baground) / normalization

    # manual_01 = (np.trapz(aanorm[:, from1:to], axis=1) + np.trapz(aanorm[:, from2:to2], axis=1)
    #              + np.trapz(aanorm[:, from3:to3], axis=1) + np.trapz(aanorm[:, from4:to4], axis=1)
    #              + np.trapz(aanorm[:, from5:to5], axis=1) + np.trapz(aanorm[:, from6:to6], axis=1)
    #              + np.trapz(aanorm[:, from7:to7], axis=1) + np.trapz(aanorm[:, from8:to8], axis=1))
    #
    # manual_00 = (np.trapz(bbnorm[:, from1:to]) + np.trapz(bbnorm[:, from2:to2])
    #              + np.trapz(bbnorm[:, from3:to3]) + np.trapz(bbnorm[:, from4:to4])
    #              + np.trapz(bbnorm[:, from5:to5]) + np.trapz(bbnorm[:, from6:to6])
    #              + np.trapz(bbnorm[:, from7:to7]) + np.trapz(bbnorm[:, from8:to8]))

    manual_01 = np.trapz(aanorm[:, ])
    manual_00 = np.trapz(bbnorm[:, ])

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
# pvalue1 = []
# pvalue2 = []
# pvalue3 = []
#
# bg = 20
# sigma1 = 3
# sigma2 = 5
# sigma3 = 10
# for i in range(100):
#
#      bg = bg + 5
#      bg1222.append(bg)
#      pvalue1.append(manPvalue8(3, bg, 30))
#      pvalue2.append(manPvalue8(5, bg, 30))
#      pvalue3.append(manPvalue8(10, bg, 30))
#
#
# bg1222 = np.array(bg1222)
# pvalue1 = np.array(pvalue1)
# pvalue2 = np.array(pvalue2)
# pvalue3 = np.array(pvalue3)
#
# print(pvalue1)
# print(bg1222)
#
# plt.title('Sigma VS Background')
# plt.ylabel('Sigma')
# plt.xlabel('Background')
# plt.plot(bg1222, pvalue2, label='\u03C3=5')
# plt.plot(bg1222, pvalue3, label='\u03C3=10')
# plt.plot(bg1222, pvalue1, label='\u03C3=3')
# plt.legend()
# # plt.yscale('log')
# plt.grid()
# plt.show()
#
# df1 = pd.DataFrame([bg1222, pvalue1, pvalue2, pvalue3],
#                    index=['bg', 'sigma3', 'sigma5', 'sigma10'])
# df1.to_excel("8picmanbg.xlsx")

# =======p value VS signals amp ================
bg1222 = []
pvalue1 = []
pvalue2 = []
pvalue3 = []

amp = 0
for i in range(70):

     bg1222.append(amp)
     pvalue2.append(manPvalue8(5, 200, amp))
     amp = amp + 1



bg1222 = np.array(bg1222)
pvalue1 = np.array(pvalue1)
pvalue2 = np.array(pvalue2)
pvalue3 = np.array(pvalue3)


plt.title('P-Value VS Signal Magnitude')
plt.ylabel('P-Value')
plt.yscale('log')
plt.xlabel('Signal Magnitude [Events/GeV]')
plt.plot(bg1222, pvalue2, label='\u03C3=5')
plt.legend()
plt.grid()
plt.show()


df1 = pd.DataFrame([bg1222, pvalue1, pvalue2, pvalue3],
                   index=['amp', 'sigma3', 'sigma5', 'sigma10'])
df1.to_excel("8pic_Naivefull.xlsx")




