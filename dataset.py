import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# データセットの読み込み
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
# 4列目の分類名のみを100データ分抽出
teacher_dataset = dataset.iloc[0:100, 4].values
# 分類名がヒオウギアヤメ（Iris setosa）であれば「1」異なれば「-1」
teacher_dataset = np.where(teacher_dataset=="Iris-setosa", -1, 1)
# 1行目（がく片の長さ）と3行目（花びらの長さ）のみ100データ分抽出
input_dataset = dataset.iloc[0:100,[0, 2]].values
# これで入力は準備完了！

# 一旦この状態で図形にしてみる。
# グラフプロパティ
markers = ('s', 'o', 'x', '^', 'v')
colors = ('green', 'yellow','red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(teacher_dataset))])
labels = ('setosa', 'versicolor')
# 描画処理
plt.clf()
plt.scatter(x=input_dataset[teacher_dataset == -1, 0], y=input_dataset[teacher_dataset == -1, 1], alpha=1.0, c=cmap(0), marker=markers[0], label=labels[0])
plt.scatter(x=input_dataset[teacher_dataset ==  1, 0], y=input_dataset[teacher_dataset ==  1, 1], alpha=1.0, c=cmap(1), marker=markers[1], label=labels[1])
plt.title("Sepal length and Petal length")
plt.xlabel("sepals length [cm]")#がく片の長さ
plt.ylabel("petal length [cm]")#花びらの長さ
plt.legend(loc="upper left")
plt.grid()
plt.savefig(u"dataset.png", dpi=100)