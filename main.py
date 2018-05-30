import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

#read xlsx file
df = pd.read_excel("./iris_data_bayes.xlsx")

#change data
df.columns = ["x0","x1","x2","x3","x4"]
df.replace("ｾﾄﾅ","0",inplace=True)
df.replace("ﾊﾞｰｼｸﾙ","1",inplace=True)
df.replace("ﾊﾞｰｼﾞﾆｶ","2",inplace=True)

#x = data
X = df[["x0","x1","x2","x3"]].values

#y = label
y = df["x4"].values

#ベイズ推定
clf = GaussianNB()
#学習
clf.fit(X,y)

#test_data
t = np.array([[2,3,1,2]])

#推定結果
print(clf.predict(t))
