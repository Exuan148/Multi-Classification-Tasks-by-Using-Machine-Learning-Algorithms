#-*- coding = utf-8 -*-
#@Time : 2023-01-06 19:28
#@File : figure2.py
#@Software: PyCharm
#@Author:HanYixuan

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# START: OWN CODE
data = pd.read_csv(r"data/figure2.csv", engine='python',index_col=None)
print(data.head(5))
for i in range(8):
    newitem=np.array(data.loc[i,:].tolist())
    max_item=newitem.max()
    min_item=newitem.min()
    data.loc[i,:]=(newitem-min_item)/(max_item-min_item)
print(data.head(5))
plt.figure(figsize=(12,3))
plt.subplot(121)
plt.title("Random Forest")
plt.plot(range(1,200,10),data.loc[4,:].tolist(),linestyle='-.')
plt.plot(range(1,200,10),data.loc[5,:].tolist(),linestyle='-.')
plt.plot(range(1,200,10),data.loc[6,:].tolist(),linestyle='-.')
plt.plot(range(1,200,10),data.loc[7,:].tolist(),linestyle='-.')
plt.xlabel("n_estimator")
plt.ylabel("acc_normal")
plt.legend(["1st","2nd","3rd","4th"])
plt.subplot(122)
plt.title("Adaboost")
plt.plot(range(1,200,10),data.loc[0,:].tolist(),linestyle='-.')
plt.plot(range(1,200,10),data.loc[1,:].tolist(),linestyle='-.')
plt.plot(range(1,200,10),data.loc[2,:].tolist(),linestyle='-.')
plt.plot(range(1,200,10),data.loc[3,:].tolist(),linestyle='-.')
plt.xlabel("n_estimator")
# plt.ylabel("acc_normal")
plt.show()
# END: OWN CODE