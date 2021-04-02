import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import xlrd

#--------------------- 8PIC-----------------------------
data = pd.read_excel(r'C:\Users\diana\Desktop\project\exel\exponant background\8peaks\8picEXPONENTunited.xlsx',
                     sheet_name='1')

# #------------   amp    -------------------
# amp_dataDriven = pd.DataFrame(data, index=[3]).to_numpy()
# amp_dataDriven = amp_dataDriven[0][1:]
# amp_dataDriven_pv = pd.DataFrame(data, index=[4]).to_numpy()
# amp_dataDriven_pv = amp_dataDriven_pv[0][1:]

amp_dataDriven1 = pd.DataFrame(data, index=[32]).to_numpy()
amp_dataDriven1 = amp_dataDriven1[0][1:]
amp_dataDriven_pv1 = pd.DataFrame(data, index=[33]).to_numpy()
amp_dataDriven_pv1 = amp_dataDriven_pv1[0][1:]

amp_naive = pd.DataFrame(data, index=[8]).to_numpy()
amp_naive = amp_naive[0][1:]
amp_naive_pv = pd.DataFrame(data, index=[9]).to_numpy()
amp_naive_pv = amp_naive_pv[0][1:]

amp_Classifier = pd.DataFrame(data, index=[13]).to_numpy()
amp_Classifier = amp_Classifier[0][1:]
amp_Classifier_pv = pd.DataFrame(data, index=[14]).to_numpy()
amp_Classifier_pv = amp_Classifier_pv[0][1:]

amp_Autoencoder = pd.DataFrame(data, index=[18]).to_numpy()
amp_Autoencoder = amp_Autoencoder[0][1:]
amp_Autoencoder_pv = pd.DataFrame(data, index=[19]).to_numpy()
amp_Autoencoder_pv = amp_Autoencoder_pv[0][1:]

amp_Autoencoder50 = pd.DataFrame(data, index=[23]).to_numpy()
amp_Autoencoder50 = amp_Autoencoder50[0][1:]
amp_Autoencoder_pv50 = pd.DataFrame(data, index=[24]).to_numpy()
amp_Autoencoder_pv50 = amp_Autoencoder_pv50[0][1:]

amp_Autoencoder250 = pd.DataFrame(data, index=[28]).to_numpy()
amp_Autoencoder250 = amp_Autoencoder250[0][1:]
amp_Autoencoder_pv250 = pd.DataFrame(data, index=[29]).to_numpy()
amp_Autoencoder_pv250 = amp_Autoencoder_pv250[0][1:]

amp_Autoencoder15 = pd.DataFrame(data, index=[37]).to_numpy()
amp_Autoencoder15 = amp_Autoencoder15[0][1:]
amp_Autoencoder_pv15 = pd.DataFrame(data, index=[38]).to_numpy()
amp_Autoencoder_pv15 = amp_Autoencoder_pv15[0][1:]





plt.title('8 Peaks With Exponential Background')
# plt.plot(amp_dataDriven, amp_dataDriven_pv, label='Event Counting Detection')
plt.plot(amp_dataDriven1, amp_dataDriven_pv1, label='Event Counting Detection')
plt.plot(amp_Classifier, amp_Classifier_pv,  label='ML: Classifier')
plt.plot(amp_Autoencoder, amp_Autoencoder_pv, label='ML: Autoencoder 100 Neurons')
plt.plot(amp_Autoencoder50, amp_Autoencoder_pv50, label='ML: Autoencoder 50 Neurons')
plt.plot(amp_Autoencoder250, amp_Autoencoder_pv250, label='ML: Autoencoder 250 Neurons')
plt.plot(amp_Autoencoder15, amp_Autoencoder_pv15, label='ML: Autoencoder 15 Neurons')
plt.plot(amp_naive, amp_naive_pv, label='Naive Event Counting Detection')
plt.legend(bbox_to_anchor=(0, 0), loc='lower left', borderaxespad=0.)
plt.grid()
plt.yscale('log')
plt.ylabel('P-value')
plt.xlabel('Signal Magnitude [Events]')
plt.show()


# #------------   BG    -------------------
# bg_dataDriven = pd.DataFrame(data, index=[37]).to_numpy()
# bg_dataDriven = bg_dataDriven[0][1:]
# bg_dataDriven_pv = pd.DataFrame(data, index=[38]).to_numpy()
# bg_dataDriven_pv = bg_dataDriven_pv[0][1:]

bg_naive1 = pd.DataFrame(data, index=[67]).to_numpy()
bg_naive1 = bg_naive1[0][1:]
bg_naive_pv1 = pd.DataFrame(data, index=[68]).to_numpy()
bg_naive_pv1= bg_naive_pv1[0][1:]

bg_naive = pd.DataFrame(data, index=[42]).to_numpy()
bg_naive = bg_naive[0][1:]
bg_naive_pv = pd.DataFrame(data, index=[43]).to_numpy()
bg_naive_pv = bg_naive_pv[0][1:]

bg_Classifier = pd.DataFrame(data, index=[47]).to_numpy()
bg_Classifier = bg_Classifier[0][1:]
bg_Classifier_pv = pd.DataFrame(data, index=[48]).to_numpy()
bg_Classifier_pv = bg_Classifier_pv[0][1:]

bg_Autoencoder = pd.DataFrame(data, index=[52]).to_numpy()
bg_Autoencoder = bg_Autoencoder[0][1:]
bg_Autoencoder_pv = pd.DataFrame(data, index=[53]).to_numpy()
bg_Autoencoder_pv = bg_Autoencoder_pv[0][1:]

bg_Autoencoder50 = pd.DataFrame(data, index=[57]).to_numpy()
bg_Autoencoder50 = bg_Autoencoder50[0][1:]
bg_Autoencoder_pv50 = pd.DataFrame(data, index=[58]).to_numpy()
bg_Autoencoder_pv50 = bg_Autoencoder_pv50[0][1:]

bg_Autoencoder250 = pd.DataFrame(data, index=[63]).to_numpy()
bg_Autoencoder250 = bg_Autoencoder250[0][1:]
bg_Autoencoder_pv250 = pd.DataFrame(data, index=[64]).to_numpy()
bg_Autoencoder_pv250 = bg_Autoencoder_pv250[0][1:]



plt.title('8 Peaks With Exponential Background')
# plt.plot(bg_dataDriven, bg_dataDriven_pv, label='Event Counting Detection')
plt.plot(bg_naive1, bg_naive_pv1, label='Event Counting Detection')
plt.plot(bg_Classifier, bg_Classifier_pv,  label='ML:Classifier')
plt.plot(bg_Autoencoder, bg_Autoencoder_pv, label='ML:Autoencoder 100 Neurons')
plt.plot(bg_Autoencoder50, bg_Autoencoder_pv50, label='ML:Autoencoder 50 Neurons')
plt.plot(bg_Autoencoder250, bg_Autoencoder_pv250, label='ML:Autoencoder 250 Neurons')
plt.plot(bg_naive, bg_naive_pv, label='Naive Event Counting Detection')
plt.legend(bbox_to_anchor=(0, 0.2), loc='lower left', borderaxespad=0.)
plt.grid()
plt.yscale('log')
plt.ylabel('P-value')
plt.xlabel('Background Magnitude [Events/GeV]')
plt.show()
