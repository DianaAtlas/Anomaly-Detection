import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import xlrd

# # --------------------- 1PEAk-----------------------------
# data = pd.read_excel(r'C:\Users\diana\Desktop\project\exel\singlepic.xlsx', sheet_name='topy1')
#
#
# # ------------   BG    -------------------
#
# bg_manual = pd.DataFrame(data,  index=[3]).to_numpy()
# bg_manual = bg_manual[0][1:71]
# manual_pv_bg = pd.DataFrame(data,  index=[4]).to_numpy()
# manual_pv_bg = manual_pv_bg[0][1:71]
#
#
# ml_bg = pd.DataFrame(data, index=[7]).to_numpy()
# ml_bg = ml_bg[0][1:16]
# ml_pv_bg = pd.DataFrame(data, index=[8]).to_numpy()
# ml_pv_bg = ml_pv_bg[0][1:16]
#
# full_bg = pd.DataFrame(data, index=[11]).to_numpy()
# full_bg = full_bg[0][1:]
# full_pv_bg = pd.DataFrame(data, index=[12]).to_numpy()
# full_pv_bg = full_pv_bg[0][1:]
#
#
# plt.title('Single Peak Known Location')
# plt.plot(bg_manual, manual_pv_bg, label='Event Counting Detection')
# plt.plot(ml_bg, ml_pv_bg,  label='ML: Classifier')
# plt.plot(full_bg, full_pv_bg, label='Naive Event Counting Detection')
# # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
# plt.legend()
# plt.grid()
# plt.yscale('log')
# plt.ylabel('P-value')
# plt.xlabel('Background Magnitude [Events/GeV]')
# plt.show()
#
# # ----------------   AMP   ---------------
# amp_manual = pd.DataFrame(data,  index=[17]).to_numpy()
# amp_manual = amp_manual[0][1:40]
# manual_pv_amp = pd.DataFrame(data,  index=[18]).to_numpy()
# manual_pv_amp = manual_pv_amp[0][1:40]
#
# ml_amp = pd.DataFrame(data, index=[22]).to_numpy()
# ml_amp = ml_amp[0][1:16]
# ml_pv_amp = pd.DataFrame(data, index=[23]).to_numpy()
# ml_pv_amp = ml_pv_amp[0][1:16]
#
# full_amp = pd.DataFrame(data, index=[26]).to_numpy()
# full_amp = full_amp[0][1:]
# full_pv_amp = pd.DataFrame(data, index=[27]).to_numpy()
# full_pv_amp = full_pv_amp[0][1:]
#
#
#
# plt.title('Single Peak Known Location')
# plt.plot(amp_manual, manual_pv_amp, label='Event Counting Detection')
# plt.plot(ml_amp, ml_pv_amp,  label='ML: Classifier')
# plt.plot(full_amp, full_pv_amp,  label='Naive Event Counting Detection')
# # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
# plt.legend()
# plt.grid()
# plt.ylabel('P-value')
# plt.yscale('log')
# plt.xlabel('Signal Magnitude [Events]')
# plt.show()
#

# ##########################################################################################################
# ##########################################################################################################
# ######################        UNKNOWN LOCATION     #######################################################
# ##########################################################################################################
# ##########################################################################################################



# ----------------   AMP   ---------------
data = pd.read_excel(r'C:\Users\diana\Desktop\project\exel\unloc.xlsx')
#
amp_manual_full = pd.DataFrame(data, index=[0]).to_numpy()
amp_manual_full = amp_manual_full[0][1:30]
amp_manual_fullpv = pd.DataFrame(data, index=[1]).to_numpy()
amp_manual_fullpv = amp_manual_fullpv[0][1:30]


amp_manual = pd.DataFrame(data,  index=[5]).to_numpy()
amp_manual = amp_manual[0][1:25]
manual_pv_amp = pd.DataFrame(data,  index=[6]).to_numpy()
manual_pv_amp = manual_pv_amp[0][1:25]


ml_amp = pd.DataFrame(data, index=[10]).to_numpy()
ml_amp = ml_amp[0][1:16]
ml_pv_amp = pd.DataFrame(data, index=[11]).to_numpy()
ml_pv_amp = ml_pv_amp[0][1:16]


auto_amp = pd.DataFrame(data, index=[15]).to_numpy()
auto_amp = auto_amp[0][1:]
auto_amp_pv = pd.DataFrame(data, index=[16]).to_numpy()
auto_amp_pv = auto_amp_pv[0][1:]

auto_amp10 = pd.DataFrame(data, index=[20]).to_numpy()
auto_amp10 = auto_amp10[0][1:]
auto_amp_pv10 = pd.DataFrame(data, index=[21]).to_numpy()
auto_amp_pv10 = auto_amp_pv10[0][1:]


auto_amp50 = pd.DataFrame(data, index=[25]).to_numpy()
auto_amp50 = auto_amp50[0][1:]
auto_amp_pv50 = pd.DataFrame(data, index=[26]).to_numpy()
auto_amp_pv50 = auto_amp_pv50[0][1:]

plt.title('Single Peak Unknown Location')
plt.plot(amp_manual, manual_pv_amp, label='Data Driven Detection')
plt.plot(ml_amp, ml_pv_amp,  label='ML:Classifier')
plt.plot(auto_amp50, auto_amp_pv50,  label='ML:Autoencoder 1 Neurons')
plt.plot(auto_amp, auto_amp_pv,  label='ML:Autoencoder 5 Neurons')
plt.plot(auto_amp10, auto_amp_pv10,  label='ML:Autoencoder 25 Neurons')
plt.plot(amp_manual_full, amp_manual_fullpv, label='Naive Event Counting')
plt.legend(bbox_to_anchor=(0, 0), loc='lower left', borderaxespad=0.)
plt.grid()
plt.yscale('log')
plt.ylabel('P-value')
plt.xlabel('Signal Magnitude [Events]')
plt.show()


# #------------   BG    -------------------
amp_datad = pd.DataFrame(data, index=[30]).to_numpy()
amp_datad = amp_datad[0][1:71]
amp_datad_pv = pd.DataFrame(data, index=[31]).to_numpy()
amp_datad_pv = amp_datad_pv[0][1:71]


ml_bg = pd.DataFrame(data, index=[35]).to_numpy()
ml_bg = ml_bg[0][1:]
ml_pv_bg = pd.DataFrame(data, index=[36]).to_numpy()
ml_pv_bg = ml_pv_bg[0][1:]


amp_autoenc = pd.DataFrame(data, index=[40]).to_numpy()
amp_autoenc = amp_autoenc[0][1:]
autoenc_pv = pd.DataFrame(data, index=[41]).to_numpy()
autoenc_pv = autoenc_pv[0][1:]


manfull_bg = pd.DataFrame(data, index=[45]).to_numpy()
manfull_bg = manfull_bg[0][1:]
manfull_bg_pv = pd.DataFrame(data, index=[46]).to_numpy()
manfull_bg_pv = manfull_bg_pv[0][1:]


amp_autoenc10 = pd.DataFrame(data, index=[50]).to_numpy()
amp_autoenc10 = amp_autoenc10[0][1:]
autoenc_pv10 = pd.DataFrame(data, index=[51]).to_numpy()
autoenc_pv10 = autoenc_pv10[0][1:]

amp_autoenc50 = pd.DataFrame(data, index=[55]).to_numpy()
amp_autoenc50 = amp_autoenc50[0][1:]
autoenc_pv50 = pd.DataFrame(data, index=[56]).to_numpy()
autoenc_pv50 = autoenc_pv50[0][1:]


plt.title('Single Peak Unknown Location')
plt.plot(amp_datad, amp_datad_pv, label='Data Driven Detection')
plt.plot(ml_bg, ml_pv_bg,  label='ML:Classifier')
plt.plot(amp_autoenc50, autoenc_pv50, label='ML:Autoencoder 1 Neurons')
plt.plot(amp_autoenc, autoenc_pv, label='ML:Autoencoder 5 Neurons')
plt.plot(amp_autoenc10, autoenc_pv10, label='ML:Autoencoder 25 Neurons')
plt.plot(manfull_bg, manfull_bg_pv, label='Naive Event Counting')
plt.legend(bbox_to_anchor=(1, 0.4), loc='upper right', borderaxespad=0.)
plt.grid()
plt.yscale('log')
plt.ylabel('P-value')
plt.xlabel('Background Magnitude [Events/GeV]')
plt.show()
