import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#plt.style.use("ggplot")


def set_param(theta):
    """
    theta : 初期値 (0~1)
    thetaによって，カテゴリを決定し，
    そのカテゴリのガウス分布の
    平均mu,標準偏差sigmaを設定する．
    return : mu , sigma

    #カテゴリ数は2
    """
    #平均と標準偏差を初期化
    mu = 0
    sigma = 0

    #カテゴリ
    categorical = 0

    #0.3の確率でカテゴリ1,0.7の確率でカテゴリ2
    if theta <= 0.3:
        categorical = 1
    else :
        categorical = 2

    #それぞれの値を決定
    if categorical == 1:
        mu = 0
        sigma = 2
    elif categorical == 2:
        mu = 2
        sigma = 6

    return mu , sigma


#分布の数
K = 2

#データ数
N = 1000

# #一つ目のガウス分布
mu1 = 1
sigma1 = 0.2
# #混合確率1 = 0.3
N1 = int(N*0.6)
#
# #二つ目のガウス分布
mu2 = 5
sigma2 = 1
# #混合確率1 = 0.7
N2 = int(N*0.4)
# mu = np.zeros(2)
# sigma = np.zeros(2)
#
# theta = np.random.rand()     #0-1の乱数を適当に与える（これが初期値)
# for i in range(K):
#     mu[i] , sigma[i] = set_param(theta)         #


#mu1 ,sigma1 = set_param(theta)


#サンプルデータxは300個と700個の二つの分布からなるデータ(一次元)(1000)

#x = np.concatenate([np.random.normal(mu[0], sigma[0], N1), np.random.normal(mu[1], sigma[1], N2)])
x = np.concatenate([np.random.normal(mu1, sigma1, N1), np.random.normal(mu2, sigma2, N2)])
plt.hist(x, bins=100, normed=True)
#plt.show()

# 初期値
#0から1までの範囲でK個の乱数を生成する
#負担率の初期値
#w = np.random.uniform(0,1,K)   #乱数Ver
w = [0.1,0.3]  #決め打ちVer
print(w)
#全ての和が1になるように和で割る
#w /=  w.sum()

#平均
mu = [0, 2]

#標準偏差
sigma = [2, 6]

#itaはN行K列　＝　1000行2列
ita = np.zeros([N, K])

# EM アルゴリズム
training_iter = 13
for epoch in range(training_iter):
    ita_sum = 0
    # Estep
    #分子をまず計算する
    for k in range(K):
        ita[:, k] = w[k] * stats.norm.pdf(x,mu[k], sigma[k])
    """
    #sumでitaを割る．
    #np.newaxisで縦ベクトルに変換する.
    #ややこしいが結局,行の和(総トピックの現時点での分母の和)でitaを割っているだけ
    #負担率の分母は全てのトピックに対してで，全てのデータではない
    #つまり，一回一回更新された値で和を計算する．
    """
    #ita = ita / ita.sum(1)[:, np.newaxis]
    #print(ita.shape)
    """
    #####間違い例(全データの和を取っている)#####
    for i in range(N):
        for j in range(K):
            ita_sum += ita[i][j]
    """
    #正解例(トピックの和を行毎に計算する)
    for k in range(K):
        ita_sum += ita[:,k]#これだとita_sumに単に1000個のデータを入れるだけで，計算ができない
    ita = ita / ita_sum[:,np.newaxis] #[:,np.newaxis]でベクトル化してnumpyのブロードキャストに頼る
    #print(ita.shape)


    # Mstep

    #混合係数の計算 , sum(0)で列ごとの和をだす.(行毎の和/データ数)
    #つまり，トピックごとの負担率の和　= ita.sum(0) = Nk
    Nk= ita.sum(0)
    #print(Nk.shape)
    w = Nk / N

    #分散の計算
    #itaのshapeは(1000,2) , Nkのshapeは2 , xのshapeは(1000,)
    #転置することでitaは(2,1000)になるので，xとの内積が計算できる．
    #転置したので，一行目がトピック1について，二行目がトピック2についてのデータになっている．
    #つまり，定義通り，それぞれのkについて計算できる．
    mu = ita.T.dot(x) / Nk

    #標準偏差

    for k in range(K):
        #k = 1 の時だと
        #(x-mu[1])^2　* k=1の負担率 の総和　　　を　　Nkで割る
        #norm.pdfのlocには標準偏差が入る
        #分散ではない
        sigma[k] = np.sqrt(((x - mu[k])*(x - mu[k]) * ita[:, k]).sum() / Nk[k])

    # 図示
    x_ = np.linspace(0, 8, 200)

    #平均と標準偏差の更新を表示
    print("mu1:",mu[0],"sigma1:",sigma[0])
    print("mu2:",mu[1],"sigma:",sigma[1],"\n")

    #グラフを描写
    y0 = w[0] * stats.norm.pdf(x_,mu[0], sigma[0])
    y1 = w[1] * stats.norm.pdf(x_,mu[1], sigma[1])
    plt.plot(x_, y0)
    plt.plot(x_, y1)
    plt.hist(x, bins=100, normed=True)
    title = "epoch: {}".format(epoch+1)
    plt.title(title)
    # plt.savefig("data/" + title + ".png")
    plt.show()
