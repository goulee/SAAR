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
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset

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

        # 前向传播
        for entity in graph_1.keys():
            graph_1[entity]=x[entity]
        

        # 构建图
        pre_score=[]
        for h_id, r_id, t_id in triples:
            h = graph_1[h_id]
            r_random = torch.tensor(np.random.rand(128),dtype=torch.float32)
            r=self.Bert_graph(r_random)
            t = graph_1[t_id]

            # 拼接向量
            concatenated_vector = torch.cat([h, r, t],dim=-1)
            #print(concatenated_vector.shape)
     
            input_sequence =concatenated_vector.view(1, -1)
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


#训练集
df = pd.read_csv(r'最新\data\cn15k\valid.tsv', sep='\t',header=None, index_col=None)


targets = torch.tensor(df.iloc[:, 2])  # 正样本
triples=df.iloc[:, :3].values
pro_targets = torch.tensor(df.iloc[:, 3]).view(-1).to(torch.float32)  # 置信度
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
5.分批次打包数据
'''

def create_data_loader(targets, pro_targets, edge, batch_size):
    dataset = TensorDataset(targets, pro_targets, edge,)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

data_loader = create_data_loader(targets, pro_targets, edge, batch_size=32)

'''
6.评估模型
'''

# 定义均方误差损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.Adam(SAAR_model.parameters(), lr=0.001)

# # # 训练模型
num_epochs =5
#pro_targets = pro_targets.unsqueeze(1).float()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_targets, batch_pro in data_loader:
        pos_scores = SAAR_model(graph,edge,batch_targets)
    # print(pos_scores.shape)
    # neg_scores = model(entities, relations, neg_targets, pro_neg, edge)
    # print(neg_scores.shape)
    #loss = margin_loss(pos_scores, pro_targets, margin)
    # 定义均方误差损失函数
    # print(pos_scores.shape)
    # print(pro_targets.shape)
        
        loss = criterion(pos_scores, pro_targets).to(torch.float32)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(data_loader)}")


# # 定义参数网格
# param_grid = {
#     'batch_size': [64, 128, 256, 512],
#     'learning_rate': [0.001, 0.005, 0.01],
#     'output_dim': [64, 128, 256, 512]
# }


# # 创建 GridSearchCV 对象
# grid_search = GridSearchCV(SAAR_model, param_grid, cv=5)

# # 在验证集上进行参数搜索
# grid_search.fit(targets, pro_targets)


# # 输出最佳超参数配置和性能
# print("Best parameters found:", grid_search.best_params_)
# print("Best mean reciprocal rank found:", grid_search.best_score_)