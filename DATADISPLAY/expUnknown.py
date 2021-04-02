import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import xlrd

#--------------------- EXPUNENT UNKNOWN-----------------------------
data = pd.read_excel(r'C:\Users\diana\Desktop\project\exel\exponant background\unknown location\unknownExpUnited.xlsx',
                     sheet_name='dian')

# #------------   amp    -------------------
amp_dataDriven = pd.DataFrame(data, index=[3]).to_numpy()
amp_dataDriven = amp_dataDriven[0][1:]
amp_dataDriven_pv = pd.DataFrame(data, index=[4]).to_numpy()
amp_dataDriven_pv = amp_dataDriven_pv[0][1:]

amp_naive = pd.DataFrame(data, index=[8]).to_numpy()
amp_naive = amp_naive[0][1:]
amp_naive_pv = pd.DataFrame(data, index=[9]).to_numpy()
amp_naive_pv = amp_naive_pv[0][1:]

amp_Classifier = pd.DataFrame(data, index=[13]).to_numpy()
amp_Classifier = amp_Classifier[0][3:]
amp_Classifier_pv = pd.DataFrame(data, index=[14]).to_numpy()
amp_Classifier_pv = amp_Classifier_pv[0][3:]

amp_Autoencoder = pd.DataFrame(data, index=[18]).to_numpy()
amp_Autoencoder = amp_Autoencoder[0][1:]
amp_Autoencoder_pv = pd.DataFrame(data, index=[19]).to_numpy()
amp_Autoencoder_pv = amp_Autoencoder_pv[0][1:]

amp_Autoencoder10 = pd.DataFrame(data, index=[23]).to_numpy()
amp_Autoencoder10 = amp_Autoencoder10[0][1:]
amp_Autoencoder_pv10 = pd.DataFrame(data, index=[24]).to_numpy()
amp_Autoencoder_pv10 = amp_Autoencoder_pv10[0][1:]

amp_Autoencoder50 = pd.DataFrame(data, index=[28]).to_numpy()
amp_Autoencoder50 = amp_Autoencoder50[0][1:]
amp_Autoencoder_pv50 = pd.DataFrame(data, index=[29]).to_numpy()
amp_Autoencoder_pv50 = amp_Autoencoder_pv50[0][1:]


plt.title('Unknown Location With Exponential Background')
plt.plot(amp_dataDriven, amp_dataDriven_pv, label='Event Counting Detection')
plt.plot(amp_Classifier, amp_Classifier_pv,  label='ML:Classifier')
plt.plot(amp_Autoencoder10, amp_Autoencoder_pv10, label='ML:Autoencoder 10 Neurons')
plt.plot(amp_Autoencoder50, amp_Autoencoder_pv50, label='ML:Autoencoder 50 Neurons')
plt.plot(amp_Autoencoder, amp_Autoencoder_pv, label='ML:Autoencoder 25 Neurons')
plt.plot(amp_naive, amp_naive_pv, label='Naive Event Counting Detection')
plt.legend(bbox_to_anchor=(0.6, 0.5), loc='upper right', borderaxespad=0.)
plt.grid()
plt.yscale('log')
plt.ylabel('P-value')
plt.xlabel('Signal Magnitude [Events]')
plt.show()


# #------------   BG    -------------------
bg_dataDriven = pd.DataFrame(data, index=[27]).to_numpy()
bg_dataDriven = bg_dataDriven[0][1:]
bg_dataDriven_pv = pd.DataFrame(data, index=[28]).to_numpy()
bg_dataDriven_pv = bg_dataDriven_pv[0][1:]

bg_naive = pd.DataFrame(data, index=[32]).to_numpy()
bg_naive = bg_naive[0][1:]
bg_naive_pv = pd.DataFrame(data, index=[33]).to_numpy()
bg_naive_pv = bg_naive_pv[0][1:]

bg_Classifier = pd.DataFrame(data, index=[37]).to_numpy()
bg_Classifier = bg_Classifier[0][1:]
bg_Classifier_pv = pd.DataFrame(data, index=[38]).to_numpy()
bg_Classifier_pv = bg_Classifier_pv[0][1:]

bg_Autoencoder = pd.DataFrame(data, index=[42]).to_numpy()
bg_Autoencoder = bg_Autoencoder[0][1:]
bg_Autoencoder_pv = pd.DataFrame(data, index=[43]).to_numpy()
bg_Autoencoder_pv = bg_Autoencoder_pv[0][1:]



plt.title('Unknown location with exponential Background')
plt.plot(bg_dataDriven, bg_dataDriven_pv, label=' Event Counting Detection')
plt.plot(bg_Classifier, bg_Classifier_pv,  label='ML:Classifier')
plt.plot(bg_Autoencoder, bg_Autoencoder_pv, label='ML:Autoencoder')
plt.plot(bg_naive, bg_naive_pv, label=' Naive Event Counting Detection')
plt.legend(bbox_to_anchor=(1, 0.3), loc='upper right', borderaxespad=0.)
plt.grid()
plt.yscale('log')
plt.ylabel('P-value')
plt.xlabel('Background Magnitude [Events/GeV]')
plt.show()
