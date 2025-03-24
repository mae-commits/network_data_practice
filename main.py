# 必要なライブラリのインポート
from torch_geometric.datasets import Planetoid
import os
from torch_geometric.transforms import RandomNodeSplit

# torch_geometric の Dataset としてダウンロード
if not os.path.exists("./dataset"):
    dataset = Planetoid(root="./dataset", name="Cora", split="full")
    

dataset = Planetoid(root="./dataset", name="Cora", split="full")

# ノードを学習データとテストデータに分割
node_splitter = RandomNodeSplit(
    split="train_rest",  # 分割方法
    num_splits=1,        # 分割数
    num_val=0.0,         # 検証データの割合
    num_test=0.4,        # テストデータの割合
    key="y",             # 正解データの属性名
)
splitted_data = node_splitter(dataset._data)
print(splitted_data.node_attrs())
print(splitted_data.train_mask)
print(splitted_data.test_mask)