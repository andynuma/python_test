import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def decision_topics(x):
    """
    トピックを決定する．
    確率：トピック
    -------------
    0.2 : 1
    0.5 : 2
    0.3 : 3
    """
    if 0 <= x < 0.3:
        return 1 #スポーツ
    elif 0.3 <= x <= 1:
        return 2 #経済
    # elif 0.7<= x <= 1:
    #     return 3 #政治
    else :
        return "error"


def create(seed,x):
    """
    選ばれたトピックとガウス分布を関連付ける
    seed (0~1):値によって以下の分布からサンプリング
    ガウス分布1:平均1,標準偏差0.2
    ガウス分布2:平均5,標準偏差1
    """
    topic = decision_topics(seed)
    if topic == 1:
        return np.random.normal(1,0.2)
    elif topic == 2:
        return np.random.normal(5,1)



####################データ生成#################################

#データ数
N = 1000
#カテゴリ数
K = 2

n = np.linspace(0,8,N)

#データ格納
x = []

for i in range(N):
    x.append(create(np.random.rand(),n[i]))#xにカテゴリ分布に従って生成したガウス分布の値を

plt.hist(x, bins=100, normed=True)
plt.show()


####################データ生成ここまで##############################

###################初期化#########################################
#負担率の初期値
#w = np.random.uniform(0,1,K)   #乱数Ver
w = [0.1,0.1]
#平均
mu = [0, 2]

#標準偏差
sigma = [2, 6]

#負担率を入れる配列 , gammaはN行K列　＝　1000行2列
gamma = np.zeros([N, K])

###################初期化ここまで##################################


# EM アルゴリズム
training_iter = 13
for epoch in range(training_iter):
    gamma_sum = 0
    # Estep
    #分子をまず計算する
    for k in range(K):
        gamma[:, k] = w[k] * stats.norm.pdf(x,mu[k], sigma[k])
    """
    #sumでgammaを割る．
    #np.newaxisで縦ベクトルに変換する.
    #ややこしいが結局,行の和(総トピックの現時点での分母の和)でgammaを割っているだけ
    #負担率の分母は全てのトピックに対してで，全てのデータではない
    #つまり，一回一回更新された値で和を計算する．

    #####間違い例(全データの和を取っている)#####
    for i in range(N):
        for j in range(K):
            gamma_sum += gamma[i][j]
    """
    #正解例(トピックの和を行毎に計算する)
    for k in range(K):
        gamma_sum += gamma[:,k]#これだとgamma_sumに単に1000個のデータを入れるだけで，計算ができない
    gamma = gamma / gamma_sum[:,np.newaxis] #[:,np.newaxis]でベクトル化してnumpyのブロードキャストに頼る
    #print(gamma.shape)


    # Mstep

    #混合係数の計算 , sum(0)で列ごとの和をだす.(行毎の和/データ数)
    #つまり，トピックごとの負担率の和　= gamma.sum(0) = Nk
    Nk= gamma.sum(0)
    #print(Nk.shape)
    w = Nk / N

    #分散の計算
    #gammaのshapeは(1000,2) , Nkのshapeは2 , xのshapeは(1000,)
    #転置することでgammaは(2,1000)になるので，xとの内積が計算できる．
    #転置したので，一行目がトピック1について，二行目がトピック2についてのデータになっている．
    #つまり，定義通り，それぞれのkについて計算できる．
    mu = gamma.T.dot(x) / Nk

    #標準偏差

    for k in range(K):
        #k = 1 の時だと
        #(x-mu[1])^2　* k=1の負担率 の総和　　　を　　Nkで割る
        #norm.pdfのlocには標準偏差が入る
        #分散ではない
        sigma[k] = np.sqrt(((x - mu[k])*(x - mu[k]) * gamma[:, k]).sum() / Nk[k])

    # 図示
    x_ = np.linspace(0, 8, 200)

    #平均と標準偏差の更新を表示
    #print("mu1:",mu[0],"sigma1:",sigma[0])
    #print("mu2:",mu[1],"sigma:",sigma[1],"\n")

    #グラフを描写
    y0 = w[0] * stats.norm.pdf(x_,mu[0], sigma[0])
    y1 = w[1] * stats.norm.pdf(x_,mu[1], sigma[1])
    plt.plot(x_, y0)
    plt.plot(x_, y1)
    plt.hist(x, bins=100, normed=True)
    title = "epoch: {}".format(epoch+1)
    plt.title(title)
    plt.show()
