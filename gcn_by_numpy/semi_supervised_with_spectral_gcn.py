# -*- coding: utf-8 -*-
# +
# https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0
# -

# 機器學習在圖(graph)上的任務高度的複雜，也高度的具有資料含量，這篇是第2篇文章，怎麼使用GCN在圖上做深度學習
# GCN是一種非常強大的神經網路，我們設計他來直接對圖進行操作，並擷取圖中的資訊，這裡我先幫你簡單複習一下上一次的post，你可以在下面的超連結找到上次的post
#
#
# introduction.....
#
#

# 然後有個炫砲的動畫 with training epoch
#

# # A Brief Recap

# 簡短複習(A Biref Recap)
# 我們在上篇po文中看到一個簡易的數學框架來表達GCN，
# 在一張圖$G$中有$N$個節點(node)，每個節點有$F^{0}$個特徵，則
#
# 我們有一個$N \times F^{0}$的特徵矩陣 $X$，以及一個以矩陣形式表達的圖結構
# 圖$G$ 的 鄰接矩陣 adjacency matrix $A$
#
# GCN的每一層hidden layer可以被表達為$H^{i} = f(H^{i-1}, A)$, where $H_{0} = X$以及$f$是傳播規則propagation rule，每一層layer會產生$N \times F_{i}$的feature matrix，其中每一行會是一個節點node的特徵
#

# 我們採用的progation rule為
#
# 1. $f(H^{i}, A) = \sigma(AH^{i}W^{i})$
# 2. $f(H^{i}, A) = \sigma(D^{-1}\hat{A}H^{i}W^{i})$, where $\hat{A} = A + I$, $I$為單位矩陣，$D^{-1}$為$\hat{A}$的degreee matrix的反矩陣

# 這些規則是將單個節點的特徵以他的鄰居得特徵做聚合，線性組合時呈上權重矩陣$W$以及activation function$\sigma$，現在更明確地將上述兩個傳播規則寫成
#
# $f(H^{i}, A)$ = transform(aggregate($A, H^{i}$), $W^{i}$)
#
# 就像我們在上一篇所說的，rule 1 表達的是一個節點的特徵是他的鄰居的和(sum)，而這麼做有兩個明顯的缺點
#     * 聚合操作沒有包含該節點自己的特徵
#     * 有多個鄰居的節點容易有更大的值在他的特徵表達中，這會帶來梯度爆炸，讓我們使用SGD這類的算法進行訓練。
#     
# 為了解兩個問題，rule 2 加入了自環，也就是單位矩陣$I$, $\hat{A} = A+I$
# 並且透過杜矩陣的反矩陣$D^{-1}$來進行normalization
#
# 在下面的解釋中，我們將rule 1 稱之為 sum rule, rule 2 稱之為 mean rule

# # Spectral Graph Convolution

# 近期有一篇paper(Kipf and Welling)提出了一個快速的spectral graph convolutions使用下列的傳播規則
#
# $$
# f(X, A) = \sigma(D^{-0.5}\hat{A}D^{-0.5}XW
# $$
#
# spectral rule僅在aggregation function尚有不同的選擇，但這基本上跟mean rule很像，事實上更像是一種幾何平均，我們把它拆開來看看他到底做了什麼
#
#

# ## Aggregation as a Weighted Sum
# 我們可以將Aggregation理解成加權平均，接著我們一路從sum rule, mean rule 走到 spectral rule
#
# ### The sum rule
# here, we have node $\in \{1, 2, 3, ..., i, ..., j, ... ,N\}$
#
# for $i^{th}$ node
#
# $$aggregate(A, X)_{i} = A_{i}X$$
#
# $$ = \sum _{j=1}^{N}A_{i, j}X_{j}$$
#
# 在這樣的情況下，若在$i^{th}$個node，第$j^{th}$個node並沒有與$i^{th}$相連，如此一來$A_{i, j}=0$，因此上式的意思即為僅和相連的節點有關，並相加，並且鄰居的貢獻完全來自於鄰接矩陣$A$
#
# ## The Mean Rule
#
# 這裡為了讓你有個更清楚的比較，我們暫時將自環的部分$\hat{A}$以$A$代替
#
#
# $$aggregate(A, X)_{i} = D^{-1}A_{i}X$$
#
# $$ = \sum _{k=1}^{N}D_{i, k}^{-1} \sum _{j=1}^{N}A_{i, j}X_{j}$$
#
# 由於$D$是一個對角矩陣，因此從$\sum _{k=1}^{N}$其實就是一個值而已
# 那麼上式就能夠改成
#
# $$ D_{i, i}^{-1} \sum _{j=1}^{N}A_{i, j}X_{j}$$
#
#
# 而我們又知道 $D^{-1}D$必須為$I$，反矩陣的定義，因此$D_{i,i}^{-1} = \frac{1}{D_{i,i}}$
#
# 那麼上式就能夠改成
#
# $$ \sum _{j=1}^{N} \frac{A_{i, j}}  {D_{i, i}}  X_{j}$$
#
#
# blablabla補充說明
#
# ## The Spectral Rule
#
# $$aggregate(A, X)_{i} = D^{-0.5}A_{i}D^{-0.5}X$$
#
# * 這裡並不是兩個$D$可以合併，這裡在做的是 1d vector level的inner product
#
# $$ = \sum _{k=1}^{N}D_{i, k}^{-0.5} \sum _{j=1}^{N}A_{i, j} \sum _{l=1}^{N}D_{j, l}^{-0.5} X_{j}$$
#
# * 然而我們知道$D$是度矩陣，是一個對角矩陣，每列/行加總只有一個值，因此我們有
#
#
# $$ \sum _{j=1}^{N} D_{i, i}^{-0.5} A_{i, j} D_{j, j}^{-0.5} X_{j}$$
#
#
# $$ \sum _{j=1}^{N} \frac{1}  {D_{i, i}^{0.5}} A_{i, j} \frac{1}{ D_{j, j}^{0.5} } X_{j}$$
#
# * 這裡就物理意義上就產生一件有趣的事情，當我們對$i^{th}$node進行aggregation時，我們不僅考慮了$i^{th}$mode的鄰居個數，同時在加總的過程，我們還考慮了$j^{th}$node的鄰居個數
#
# 與mean rule類似，spectral rule也對了aggregation做了一層normalization，這樣的momalization使得output feature和input feature大約在同個scale上，然而 spectral rule的aggregation會使得$i^{th}$鄰居數更少的數值更大，反之則更小，這個設計使得鄰居少的node比起鄰居多的node能夠提供更有效的資訊

# # Semi-Supervised Classification with GCNs
#
# 在半監督式學習中我們希望有標籤的資料以及沒有標籤的資料都能夠進行訓練，而截至為此的討論我們都假設整張圖結構都是可取得的，並且有些節點我們有標籤，有些沒有，這稱作為(transductive setting)，我們也能夠把這樣的任務稱作為node prediction
#
# [Transduction (machine learning)](https://en.wikipedia.org/wiki/Transduction_(machine_learning))
#
# 在前面的規則我們可以看到，我們透過鄰居來將節點去合起來，因此鄰居們容易有更相似的特徵表現，這樣的圖性質也被稱作[homophily](https://en.wikipedia.org/wiki/Homophily)，這樣的性質在許多現實世界的網路也都存在，特別是社交網路
#
# 在我們前一篇貼文中可以看到，在使用圖結構的情況下，甚至我們只是隨機初始化GCN的Weights，就能夠有很好的切分性，我們能夠將這樣的特性使用在GCN的訓練當中，讓沒有標籤的資料也能夠因為圖結構帶來一定的資訊量
#
# 透過傳播具有標籤的節點的資訊量到不具有標籤的節點可以這麼做
#
# 1. 做GCN的前項傳播(這一項就已經讓沒有標籤的node也貢獻的資訊量，透過鄰接矩陣$A$)
# 2. 在最後一層的GCN以每一列為基準做sigmoid
# 3. 在已經知道標籤的資料上計算cross entropy
# 4. 反向傳播，更新weights
#

# # Community Prediction in Zachary's Karate Club
#
# * 安裝mxnet
# * check the notebook


