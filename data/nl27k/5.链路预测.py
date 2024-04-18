from collections import defaultdict
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch.optim as optim

class SAAR(nn.Module):
    def __init__(self, num_nodes,input_dim,hidden_dim,output_dim,input_size,hidden_size):
        # 超参数
        super(SAAR, self).__init__()

        #GNN参数
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim=hidden_dim
        self.output_dim = output_dim

        #LSTM参数
        self.input_size=input_size, 
        self.hidden_size=hidden_size,
        #self.num_layers=num_layers,

        #bert模型配置
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    # 或者选择其他预训练的BERT模型
        self.BertModel = BertModel.from_pretrained('bert-base-uncased')

        #GNN模型配置
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        #LSTM模型配置
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    "2.Bert模型"
    # 加载BERT tokenizer和模型

    def Bert_graph(self,sentence):   
        # 使用tokenizer将句子转换为token IDs
        #input_ids = tokenizer.encode(sentence, add_special_tokens=True)
        #将token IDs转换为PyTorch张量
        if len(sentence)==1:
            sentence.append(0)
            input_ids_tensor = torch.tensor([sentence])
        else:
            input_ids_tensor = torch.tensor([sentence])

        # 获取BERT模型的输出
        with torch.no_grad():
            outputs =self.BertModel(input_ids_tensor)

        # 获取CLS标记对应的隐藏状态
        cls_embedding = outputs.pooler_output

        # 将PyTorch张量转换为NumPy数组
        cls_embedding_np = cls_embedding.numpy()[0]
        return(cls_embedding_np)

    def forward(self,graph,edge,triples):
        #bert模型
        graph_1={}
        for entity, neighbors in graph.items():
            if len(neighbors)<=512:
                graph_1[entity]=self.Bert_graph(neighbors)
            else:
                #超过512的序列要使用random.sample进行随机截取
                random_sample = random.sample(neighbors, 512)
                graph_1[entity]=self.Bert_graph(random_sample)
        # print(type(graph_1))
        # prnit()
        # np.save('graph_dict.npy', graph_1)  # 保存
    
        #graph= np.load('graph_dict.npy', allow_pickle=True, encoding='bytes').tolist()
        y=list(graph_1.values())
        #print(y)
        #print(list(graph_1.values())[0])

        x = torch.tensor(y,dtype=torch.float32)# 示例节点特征
        data = Data(x=x, edge_index=edge)
        x, edge_index = data.x, data.edge_index

        # 第一层GCN卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层GCN卷积
        x = self.conv2(x, edge_index)
        #print(x)

        # 前向传播
        for entity in graph_1.keys():
            graph_1[entity]=x[entity]
        
        #np.save('graph_update.npy', graph)  # 保存

        # 构建图
        pre_score=[]
        for h_id, r_id, t_id in triples:
            h = graph_1[h_id]
            r = torch.tensor(np.random.rand(60),dtype=torch.float32)
            t = graph_1[t_id]

            # 拼接向量
            concatenated_vector = torch.cat([h, r, t],dim=-1)
            #print(concatenated_vector.shape)
     
            input_sequence =concatenated_vector.view(1, -1)
            #print(input_sequence.dtype)
            #print(input_sequence.size(-1))
            # 前向传播

            #lstm = nn.LSTM(input_size=input_sequence.size(-1), hidden_size=128, num_layers=1, batch_first=True)
            output, (hidden_state, cell_state) =self.lstm(input_sequence)
            #print(output)

            # 输出更新后的向量
            score_distance = torch.sum(output)
            #print( score_distance)
            score_weight = torch.sigmoid(score_distance)
            pre_score.append(score_weight.item())
        pre_score=torch.tensor(pre_score,requires_grad=True)
        return pre_score


# '''
# 1.预处理数据
# '''
    
#总的数据集
df_total=pd.read_csv(r'最新\data\cn15k\data.tsv', sep='\t',header=None, index_col=None)
triples_total=df_total.iloc[:, :3].values




edge = torch.tensor([df.iloc[:, 0], df.iloc[:, 2]],dtype=torch.long)  # 读取边，用于卷积网络



'''
2.超参数定义

'''
num_nodes = 15000
input_dim = 768
hidden_dim = 10
output_dim = 128
input_size=316
hidden_size=128
#num_layers=1

'''
3.定义模型

'''

SAAR_model=SAAR(num_nodes,input_dim,hidden_dim,output_dim,input_size,hidden_size)

'''
4.采样邻居实体
'''
def traverse_triplets(triplets):
    graph = defaultdict(list)

    # 构建图
    for entity1, relation, entity2 in triplets:
        graph[entity1].append(entity2)
        graph[entity2].append(entity1)
    return graph
graph=traverse_triplets(triples_total)



'''

链路预测
'''


'''
3.测试模型数据

'''

#加载保存的模型的权重和参数
SAAR_model.load_state_dict(torch.load('CTransR\data\cn15k\SAAR_model.pth'))

#设置模型为评估模式
SAAR_model.eval()

# 测试数据（示例数据，根据您的数据集进行替换）
test_data = pd.read_csv(r'最新\data\cn15k\test.tsv', sep='\t',header=None, index_col=None)


targets_test = torch.tensor(test_data.iloc[:, 2])  # 正样本
#relations = torch.tensor(df.iloc[:, 1])
triples_test=test_data.iloc[:, :3].values
pro_targets_test = torch.tensor(test_data.iloc[:, 3]).view(-1).to(torch.float32)  # 置信度
edge_test = torch.tensor([test_data.iloc[:, 0], test_data.iloc[:, 2]],dtype=torch.long)  # 读取边，用于卷积网络


pos_scores = SAAR_model(graph,edge,targets_test)



def compute_mrr(predictions, targets):
    mrr = 0.0
    for pred, target in zip(predictions, targets):
        if target in pred:
            rank = pred.index(target) + 1
            mrr += 1.0 / rank
    mrr /= len(predictions)
    return mrr

def compute_hits_at_n(predictions, targets, n):
    hits = np.zeros(len(predictions[0]))
    for pred, target in zip(predictions, targets):
        if target in pred[:n]:
            hits[n-1] += 1
    hits /= len(predictions)
    return hits

# 示例数据，predictions 和 targets 是列表，每个元素是一个样本的预测值和实际值
predictions = [[1, 3, 2, 4], [2, 1, 3, 4], [3, 2, 1, 4]]
targets = [3, 2, 1]

# 计算 MRR
mrr = compute_mrr(predictions, targets)
print("MRR:", mrr)

# 计算 Hits@N
n = 3
hits_at_n = compute_hits_at_n(predictions, targets, n)
print("Hits@{}:".format(n), hits_at_n)
















'''
4.MRR
'''
def mean_reciprocal_rank(predictions):
    mrr = 0
    for pred_list in predictions:
        found = False
        for i, pred in enumerate(pred_list[1:], 1):
            if pred == pred_list[0]:
                mrr += 1 / i
                found = True
                break
        if not found:
            mrr += 0  # 若未找到，则增加零
    return mrr / len(predictions)

'''
5.Hit@K
'''

def hit_at_k(predictions, k):
    hits = 0
    for pred_list in predictions:
        if pred_list[0] in pred_list[1:k+1]:
            hits += 1
    return hits / len(predictions)


# # 测试数据（示例数据，根据您的数据集进行替换）
test_data = pd.read_csv(r'最新\data\cn15k\test.tsv', sep='\t',header=None, index_col=None)
# 将 DataFrame 转换为列表形式，方便处理
data = test_data.values.tolist()

# 根据每个查询对数据进行分组
grouped_data = {}
for entry in data:
    query = tuple(entry[:3])  # 使用前三列作为查询键
    if query not in grouped_data:
        grouped_data[query] = []
    grouped_data[query].append(entry[3])  # 将第四列添加到查询的值列表中

# 转换为预测列表
predictions = []
for query, targets in grouped_data.items():
    predictions.append([targets[0]] + targets[1:])  # 将第一个目标作为正确预测，其余作为其他预测

# 计算指标
mrr_value = mean_reciprocal_rank(predictions)
hit_at_1_value = hit_at_k(predictions, 1)
hit_at_3_value = hit_at_k(predictions, 3)
hit_at_10_value = hit_at_k(predictions, 10)

# 打印结果
print("MRR:", mrr_value)
print("Hit@1:", hit_at_1_value)
print("Hit@3:", hit_at_3_value)
print("Hit@10:", hit_at_10_value)



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# 加载模型
model = torch.load('your_model.pth')  # 请替换成你保存的模型文件路径
model.eval()  # 将模型设置为评估模式

# 准备数据
# 假设有一个 DataLoader 加载了测试数据
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义评估指标的函数
def evaluate_model(model, data_loader):
    mrr_total = 0.0
    hits_at_k_total = np.zeros(3)  # 在这里我们计算 Hit@1, Hit@3 和 Hit@5
    total_samples = 0

    with torch.no_grad():
        for features, labels in data_loader:
            # 进行预测
            outputs = model(features)
            # 计算 MRR
            _, predicted_ranks = outputs.sort(descending=True)
            for i, rank in enumerate(predicted_ranks):
                true_label = labels[i]
                reciprocal_rank = 1 / (list(rank).index(true_label) + 1)
                mrr_total += reciprocal_rank

                # 计算 Hit@K
                for k in range(1, 4):
                    if true_label in rank[:k]:
                        hits_at_k_total[k-1] += 1

            total_samples += len(labels)

    mrr = mrr_total / total_samples
    hits_at_k = hits_at_k_total / total_samples

    return mrr, hits_at_k

# 计算评估指标
mrr, hits_at_k = evaluate_model(model, test_data_loader)
print("MRR:", mrr)
print("Hits@1:", hits_at_k[0])
print("Hits@3:", hits_at_k[1])
print("Hits@5:", hits_at_k[2])
