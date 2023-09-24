#-*- coding = utf-8 -*-
#@Time : 2023-01-03 21:11
#@File : DataPreprocessing.py
#@Software: PyCharm
#@Author:HanYixuan

import pandas as pd
import numpy as np

# START: OWN CODE
def Preprocessing(data_name,method=None):
    print("getting data {data_name}, method={method}".format(data_name=data_name, method=method))
    if data_name=="Algerian_Forest_Fire":
        data = pd.read_csv(r"data/Algerian_Forest_Fire/Algerian_forest_fires_dataset_UPDATE.csv", engine='python', index_col=None)
        for i,col_v in enumerate(data.columns.values):
            data.columns.values[i]=col_v.strip(' ')

        data["FWI"] = data["FWI"].fillna(data["FWI"].mean())
        data.drop(["year"], inplace=True, axis=1)
        area_labels = data["area"].unique().tolist()
        data["area"] = data["area"].apply(lambda x: area_labels.index(x))
        for i,class_nm in enumerate(data["Classes"]):
            data.loc[i,"Classes"]=class_nm.strip(' ')
        class_labels = data["Classes"].unique().tolist()
        data["Classes"] = data["Classes"].apply(lambda x: class_labels.index(x))
        x = data.iloc[:, data.columns != "Classes"]
        y = data.iloc[:, data.columns == "Classes"]

        if method is not None:
            if method == "standard_scale":
                mu = np.mean(x, axis=0)
                sigma = np.std(x, axis=0)
                x = (x - mu) / sigma
            elif method=="normal_scale":
                d=np.max(x,axis=0)-np.min(x,axis=0)
                x=(x-np.min(x,axis=0))/d
            else:
                print("no such method")
                return

        feature_name=x.columns.values
        return x,y,feature_name,class_labels

    elif data_name == "Algerian_Forest_Fire_withoutFWI":
        data = pd.read_csv(r"data/Algerian_Forest_Fire/Algerian_forest_fires_dataset_UPDATE_withoutFWI.csv",
                               engine='python', index_col=None)
        for i, col_v in enumerate(data.columns.values):
                data.columns.values[i] = col_v.strip(' ')

        data.drop(["year"], inplace=True, axis=1)
        area_labels = data["area"].unique().tolist()
        data["area"] = data["area"].apply(lambda x: area_labels.index(x))
        for i, class_nm in enumerate(data["Classes"]):
                data.loc[i, "Classes"] = class_nm.strip(' ')
        class_labels = data["Classes"].unique().tolist()
        data["Classes"] = data["Classes"].apply(lambda x: class_labels.index(x))
        x = data.iloc[:, data.columns != "Classes"]
        y = data.iloc[:, data.columns == "Classes"]

        if method is not None:
            if method == "standard_scale":
                mu = np.mean(x, axis=0)
                sigma = np.std(x, axis=0)
                x = (x - mu) / sigma
            elif method=="normal_scale":
                d=np.max(x,axis=0)-np.min(x,axis=0)
                x=(x-np.min(x,axis=0))/d
            else:
                print("no such method")
                return

        feature_name = x.columns.values
        return x, y, feature_name, class_labels

    elif data_name == ("Wine_Quality_Red" or "Wine_Quality_White"):
        data = pd.read_csv(r"data/Wine_Quality/{data_name}.csv".format(data_name=data_name), engine='python',
                               index_col=None)
        for i, col_v in enumerate(data.columns.values):
            data.columns.values[i] = col_v.strip(' ')

        index_labels=data["quality"].unique().tolist()
        class_labels = ['quality'+str(i) for i in index_labels]
        data["quality"] = data["quality"].apply(lambda x: index_labels.index(x))
        x = data.iloc[:, data.columns != "quality"]
        y = data.iloc[:, data.columns == "quality"]

        if method is not None:
            if method == "standard_scale":
                mu = np.mean(x, axis=0)
                sigma = np.std(x, axis=0)
                x = (x - mu) / sigma
            elif method == "normal_scale":
                d = np.max(x, axis=0) - np.min(x, axis=0)
                x = (x - np.min(x, axis=0)) / d
            else:
                print("no such method")
                return

        feature_name = x.columns.values
        return x, y, feature_name, class_labels

    elif data_name=="Credit":
        data = pd.read_csv(r"data/Credit/Credit.csv", engine='python',
                           index_col=None)
        for i, col_v in enumerate(data.columns.values):
            data.columns.values[i] = col_v.strip(' ')

        data = data.dropna()
        class_labels=data["Classes"].unique().tolist()
        for f in ["A1","A4","A5","A6","A7","A9","A10","A12","A13","Classes"]:
            Ax_labels = data[f].unique().tolist()
            data[f] = data[f].apply(lambda x: Ax_labels.index(x))

        x = data.iloc[:, data.columns != "Classes"]
        y = data.iloc[:, data.columns == "Classes"]

        if method is not None:
            if method == "standard_scale":
                mu = np.mean(x, axis=0)
                sigma = np.std(x, axis=0)
                x = (x - mu) / sigma
            elif method == "normal_scale":
                d = np.max(x, axis=0) - np.min(x, axis=0)
                x = (x - np.min(x, axis=0)) / d
            else:
                print("no such method")
                return

        feature_name = x.columns.values
        return x, y, feature_name, class_labels
# END: OWN CODE
