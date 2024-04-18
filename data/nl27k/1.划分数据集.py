
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv(r'最新\data\nl27k\data.tsv', delimiter='\t')

# 划分数据
train_data, temp_data = train_test_split(data, test_size=0.15, random_state=42)
test_data, valid_data = train_test_split(temp_data, test_size=0.47, random_state=42)


# # 打印数据集的大小
print(f"训练集大小：{len(train_data)}")
print(f"验证集大小：{len(valid_data)}")
print(f"测试集大小：{len(test_data)}")


# 保存划分后的数据
train_data.to_csv(r'最新\data\nl27k\train.tsv', sep='\t', index=False)
test_data.to_csv(r'最新\data\nl27k\test.tsv', sep='\t', index=False)
valid_data.to_csv(r'最新\data\nl27k\valid.tsv', sep='\t', index=False)

