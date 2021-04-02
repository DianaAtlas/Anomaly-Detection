
import numpy as np
import scipy.stats as stats
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
# ============================================================
# =================creating DATA==============================
# ============================================================


def DataDrivenMass(mu):
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

    exp = 200 * np.exp(-indexv / 90)
    Amp = 150
    mu = mu
    sigma_training = 5
    numofpredictions = 60000
    normalization = 2 * np.sqrt(exp)

    # ============================================================
    # =================creating TEST data=========================
    # ============================================================

    gausaray = []
    for _ in range(numofpredictions):
       x = gaussian_generator(mu, sigma_training, Amp, exp)
       gausaray.append(x)

    aa, bb = fluctuation_manufacture(gausaray, exp)
    aanormal = (np.array(aa) - exp) / normalization
    bbnormal = (np.array(bb) - exp) / normalization

    conv_index = np.linspace(0, 3*sigma_training,  3*sigma_training)
    conv_window = 0.5 * (Amp * (stats.norm.pdf(conv_index, (3*sigma_training+1)/2, sigma_training)
                                + stats.norm.pdf(conv_index + 1, (3*sigma_training+1)/2, sigma_training)))
# for data with signal
    old_shape = aanormal.shape
    aanormal = np.array(aanormal).flatten('c')
    index_conv = np.convolve(aanormal, conv_window, mode='same')
    index_conv = np.reshape(index_conv, old_shape)
    index_array = np.argmax(index_conv,  axis=1)
    aanormal = aanormal.reshape(old_shape)

    print('index_cov',index_conv)
    print('mass', mass)

# for data without signal
    old_shape_bbnormal = bbnormal.shape
    bbnormal = np.array(bbnormal).flatten('c')
    index_convb = np.convolve(bbnormal, conv_window, mode='same')
    index_convb = np.reshape(index_convb, old_shape_bbnormal)
    index_arrayb = np.argmax(index_convb,  axis=1)
    bbnormal = bbnormal.reshape(old_shape_bbnormal)


    manual_01 = []
    manual_00 = []
    manfull_01 = []
    manfull_00 = []

    for i in range(60000):
        froma = int(index_array[i] - 1.5*sigma_training)
        toa = int(index_array[i] + 1.5*sigma_training)

        fromb = int(index_arrayb[i] - 1.5*sigma_training)
        tob = int(index_arrayb[i] + 1.5 * sigma_training)

        manual_01.append(np.trapz(aanormal[i][froma:toa]))
        manual_00.append(np.trapz(bbnormal[i][fromb:tob]))
        # manfull_01.append(np.trapz(aanormal[i]))
        # manfull_00.append(np.trapz(bbnormal[i]))


    manualmedian_00 = np.median(manual_01)
    manualsum_00 = sum_all_values_over_manual(manual_00, manualmedian_00)
    print('\033[1m', 'manual calc', '\033[0m')
    print('value of mass is', mu)
    print('presantage', manualsum_00 / numofpredictions, '+-', math.sqrt(manualsum_00 / numofpredictions))
    pvalue = manualsum_00 / numofpredictions

    return pvalue

# ======== p value VS background ===============
#
bg1222 = []
pvalue2 = []

mass = 10
for i in range(9):
     bg1222.append(mass)
     pvalue2.append(DataDrivenMass(mass))
     mass = mass + 10

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
df1.to_excel("DATAdrivenMASS111.xlsx")


