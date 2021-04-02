import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import xlrd

#--------------------- 8PIC-----------------------------
data = pd.read_excel(r'C:\Users\diana\Desktop\project\exel\8pic.xlsx')

# #------------   amp    -------------------
amp_dataDriven = pd.DataFrame(data, index=[2]).to_numpy()
amp_dataDriven = amp_dataDriven[0][1:]
amp_dataDriven_pv = pd.DataFrame(data, index=[3]).to_numpy()
amp_dataDriven_pv = amp_dataDriven_pv[0][1:]


amp_Classifier = pd.DataFrame(data, index=[6]).to_numpy()
amp_Classifier = amp_Classifier[0][1:]
amp_Classifier_pv = pd.DataFrame(data, index=[7]).to_numpy()
amp_Classifier_pv = amp_Classifier_pv[0][1:]


plt.title('8 Peaks With Constant Background')
plt.plot(amp_dataDriven, amp_dataDriven_pv, label='Event Counting Detection')
plt.plot(amp_Classifier, amp_Classifier_pv,  label='ML: Classifier')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
plt.grid()
plt.yscale('log')
plt.ylabel('P-value')
plt.xlabel('Signal Magnitude [Events]')
plt.show()


# #------------   BG    -------------------
bg_dataDriven = pd.DataFrame(data, index=[13]).to_numpy()
bg_dataDriven = bg_dataDriven[0][1:]
bg_dataDriven_pv = pd.DataFrame(data, index=[14]).to_numpy()
bg_dataDriven_pv = bg_dataDriven_pv[0][1:]


bg_Classifier = pd.DataFrame(data, index=[17]).to_numpy()
bg_Classifier = bg_Classifier[0][1:]
bg_Classifier_pv = pd.DataFrame(data, index=[18]).to_numpy()
bg_Classifier_pv = bg_Classifier_pv[0][1:]


plt.title('8 Peaks With Constant Background')
plt.plot(bg_dataDriven, bg_dataDriven_pv, label='Event Counting Detection')
plt.plot(bg_Classifier, bg_Classifier_pv,  label='ML:Classifier')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
plt.grid()
plt.yscale('log')
plt.ylabel('P-value')
plt.xlabel('Background Magnitude [Events/GeV]')
plt.show()
