#分散が与えられた時，平均をベイズ推定で求める

import numpy as np
import matplotlib.pyplot as plt
import argparse

def create_data(mu,sigma):
    """
    平均mu,分散sigmaの正規分布を生成
    """
    return np.random.normal(mu,sigma)

#パサー作成
parser = argparse.ArgumentParser()

#引数の追加
parser.add_argument("-n","-number",type=int, nargs=1)

#引数の解析
args = parser.parse_args()
#初期化

#事前文布の初期化
#平均1,分散3,(標準偏差sqrt(3))
mu0 = 1
print("mu0:",mu0)
sigma0 = 3
N = args.n[0]
n = np.linspace(-5,5,N)
data = []

#ガウス分布
for i in range(len(n)):
    data.append(create_data(mu0,sigma0))
plt.hist(data,bins = 100)
plt.show()
# print(data)
sigma = sigma0

#muの推定（PRMLの式そのまま）
mu_ml = np.mean(data)
mu_n = (sigma*mu0 + N * sigma0*mu_ml)/(N * sigma0 + sigma )
sigma_n = (sigma0*sigma) / (sigma + N * sigma0)


print("mu_n:",mu_n)
#print("sigma_n:",sigma_n)
