import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import xlrd

#--------------------- 1PIC-----------------------------
data = pd.read_excel(r'C:\Users\diana\Desktop\supcon\ALL0007.CSV')
x= pd.DataFrame(data,  index=[1]).to_numpy()

x = .x[:]
print(x)