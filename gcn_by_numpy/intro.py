# -*- coding: utf-8 -*-
# https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
# reference


import numpy as np
import networkx

# 鄰接矩陣，表示了一個圖結構，有N=4個節點
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)

# 特徵矩陣，每個節點實質上為一筆資料，具有特徵X_i
# 因此特徵矩陣會是一個Nodes * features的矩陣 N x d
# 這裡N = 4, d = 2
X = np.matrix([
    [i, -i]
    for i in range(A.shape[0])
], dtype=float)
X


# 鄰接矩陣定義了傳遞關係，該鄰接矩陣為一個有像圖，可以參考Reference
# 舉例，節點0的鄰接矩陣為 0 1 0 0 表示經過一次操作 X' = AX
# 節點的資訊會變成鄰接節點的總和
# graph concolutional layer表示每個節點會變成他和他的臨街的agreegation
# node 0 現在為node 1
# node 1 現在為 node 2 + node 3
# node 2 現在為 node 1
# node 3 現在為 node 0 + node 2
A * X

# +
# 這麼做並沒有考慮到自身的特徵，因此我們要加入Identity matrix I
# 連接多個neighbor的node會導致大量的輸入，之後會梯度爆炸，aggregation應該要做一個normalization
# 有算數平均數以及幾何平均數

# 現在的A_hat考慮了自己的特徵(也稱作自環)
I = np.matrix(np.eye(A.shape[0]))
A_hat = A + I
A_hat * X
# -

# 這裡採用D-1 度矩陣的反矩陣作為normalization，實質上就是算術平均數
# 我們也可以看到只要有圖的鄰接矩陣定義，我們就能夠自動計算D以及D-1
D = np.array(np.sum(A, axis=0))[0]
D = np.matrix(np.diag(D))
D

# +
# f(X, A) = D-1AX

# 從這裡來看，做aggregation時以每一個row操作，加起來都是1
D**-1 * A
# -

D**-1 * A * X

D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))
D_hat ** -1 * A

# +
# 現在我們加入feature weights W
# 整個操作 f(X, A, W) = D-1 A_hat X W
# 即 特徵做權重之後再根據圖來進行更新
W = np.matrix([
    [1, -1],
    [-1, 1]
])

D_hat**-1 * A_hat * X * W
# -

# 如果aggregation結束我們只想要有一個features
# 那麼W的dimension就設為1即可
# 也就是說，W的dimension是 d_in x d_out
W = np.matrix([
    [1],
    [-1]
])
D_hat**-1 * A_hat * X * W


# +
# 加入activatation function

def relu(x):  # 定義 ReLU 函數
    return np.maximum(x, 0)


W = np.matrix([
    [1, -1],
    [-1, 1]
])

relu(D_hat**-1 * A_hat * X * W)
# -

# # Zachary Karate club

# +
# 一個社群網路資料，節點代表俱樂部會員
# 邊代表會員的相互關係
# 中間有一些糾葛，最後分成A和I兩群
from networkx import karate_club_graph, to_numpy_matrix
zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())

A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))


# +
import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph(A)

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
nx.draw(G, with_labels=True, ax=ax)
plt.show()

# +
# 定義GCN的權重


W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))  # 維度=(節點數, 特徵數)
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2))  # 最後要分兩群，所以維度定為(前一層特徵數, 2)

print(A.shape, A_hat, D_hat, sep='\n')
print('-' * 60)
print(W_1.shape, W_2.shape)  # 第一層hidden layer 4, 第二層input layer 2


# +
# input feature為identity?
def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)


H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2
output
# -

# 將output存在一個dict，映射回去原本的node
feature_representations = {
    node: np.array(output)[node]
    for node in zkc.nodes()}

# +
# 接著看圖，voila!
