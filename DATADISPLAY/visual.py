import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import xlrd

#--------------------- 1PIC-----------------------------
data = pd.read_excel(r'C:\Users\diana\Desktop\project\exel\exponant background\massun.xlsx')

# #------------   mass  -------------------
mass_event = pd.DataFrame(data,  index=[0]).to_numpy()
mass_event = mass_event[0][1:]
mass_event_pv = pd.DataFrame(data,  index=[1]).to_numpy()
mass_event_pv = mass_event_pv[0][1:]


datadriven = pd.DataFrame(data, index=[5]).to_numpy()
datadriven = datadriven[0][1:]
datadriven_pv = pd.DataFrame(data, index=[6]).to_numpy()
datadriven_pv = datadriven_pv[0][1:]


naieve = pd.DataFrame(data, index=[30]).to_numpy()
naieve = naieve[0][1:]
naieve_pv = pd.DataFrame(data, index=[31]).to_numpy()
naieve_pv = naieve_pv[0][1:]


classifier = pd.DataFrame(data, index=[10]).to_numpy()
classifier = classifier[0][1:]
classifier_pv = pd.DataFrame(data, index=[11]).to_numpy()
classifier_pv = classifier_pv[0][1:]


autoencoder25 = pd.DataFrame(data, index=[15]).to_numpy()
autoencoder25 = autoencoder25[0][1:]
autoencoder25_pv = pd.DataFrame(data, index=[16]).to_numpy()
autoencoder25_pv = autoencoder25_pv[0][1:]

autoencoder50 = pd.DataFrame(data, index=[20]).to_numpy()
autoencoder50 = autoencoder50[0][1:]
autoencoder50_pv = pd.DataFrame(data, index=[21]).to_numpy()
autoencoder50_pv = autoencoder50_pv[0][1:]

autoencoder10 = pd.DataFrame(data, index=[25]).to_numpy()
autoencoder10= autoencoder10[0][1:]
autoencoder10_pv = pd.DataFrame(data, index=[26]).to_numpy()
autoencoder10_pv = autoencoder10_pv[0][1:]


plt.title(' Single Peak Exponential Background ')
plt.plot(mass_event, mass_event_pv,'--', label='Event Counting Detection')
plt.plot(datadriven, datadriven_pv, label='Data Driven Detection')
plt.plot(naieve, naieve_pv, label='Naive Event Counting')
plt.plot(classifier, classifier_pv, '--', label='ML: Classifier')
plt.plot(autoencoder25, autoencoder25_pv, label='ML: Autoencoder 25')
plt.plot(autoencoder10, autoencoder10_pv, label='ML: Autoencoder 5')
plt.plot(autoencoder50, autoencoder50_pv, label='ML: Autoencoder 1')
plt.legend()
plt.grid()
plt.ylabel('p-value')
plt.xlabel('M[GeV]')
plt.yscale('log')
plt.show()
