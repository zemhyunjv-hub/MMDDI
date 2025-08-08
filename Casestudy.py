#!/usr/bin/env python
# coding: utf-8

# In[101]:
import csv
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import math
import argparse

from sklearn.model_selection import KFold


import sys
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from radam import RAdam
import torch.nn.functional as F

from ConOperation import perturb_zeros_to_ones, shuffle_random_segment
# from MINE_test import MINE
from distangling2_test import mutual_information_loss
from metrics_test import Metric
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
import os

# In[102]:

seed = 0
random.seed(seed)
# 设置环境变量PYTHONHASHSEED。这个环境变量用于控制Python中dict和set的哈希随机化行为
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='small',choices=['small','big'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--bs', type=int, help= 'batch_size', default=128)
    parser.add_argument('--o', type=str, help= 'output the specific drug-pairs scores to make case study', default= 'train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=2e-5, help= 'learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('--topk',type=int, default=4, help='number of topk')
    parser.add_argument('--visualization', action='store_true', help='enable visualization (default: False)')
    parser.add_argument("--c", action='store_true', help='enable case study (default: False)') 
    args = parser.parse_args()
    return args

args = parse_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device状态main：",device)
print('--- Data Preparation ---')
df_drug = pd.read_csv('./data/DrugInfo_540.csv')
extraction = pd.read_csv('./data/extractionDDI_34268.csv').iloc[:1000,:]
extraction.drop(columns=['drugA_smiles','drugB_smiles'], inplace=True)
drug_num = 540
# 将药物名称映射到索引
drug_to_index = {drug: idx for idx, drug in enumerate(df_drug['name'].tolist())}
print("药物数量: {} / 572".format(drug_num))
print("药物交互数量: {} / 37264".format(len(extraction)))

con_loss_T = 0.05
event_num=65
learn_rating = args.lr
weight_decay_rate = args.weight_decay
topk=args.topk
batch_size = args.bs
epo_num = args.epochs
# batch_size=512 # 文章原本的设置
# epo_num=120

motiv_num = 2
bert_n_heads = 4
bert_n_layers = 2
drop_out_rating = 0.4
cross_ver_tim = 5
feature_list = ["smile", "target", "enzyme", "pathway"]
# In[103]:
def prepare(feature_list):
    df = pd.DataFrame()  # 原始多热向量嵌入表
    embedding_table1 = np.zeros((drug_num, 0), dtype=float)  # 基于结构相似性的嵌入表
    for feature in feature_list:
        # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
        embedding_table1 = np.hstack((embedding_table1, feature_vector(feature, df_drug)))  # 1258*1258
        all_feature = []
        drug_list = np.array(df_drug[feature].tolist())
        for i in drug_list:
            for each_feature in i.split('|'):
                if each_feature not in all_feature:
                    all_feature.append(each_feature)  # obtain all the features
        feature_matrix = np.zeros((drug_num, len(all_feature)), dtype=np.float32)
        df_feature = pd.DataFrame(feature_matrix,
                                  columns=all_feature)  # Consrtuct feature matrices with key of dataframe
        for i in range(drug_num):
            for each_feature in df_drug[feature].iloc[i].split('|'):
                df_feature[each_feature].iloc[i] = 1
        df = pd.concat([df, df_feature], axis=1)

    embedding_table2 = np.array(df, dtype=np.float32)  # (540, 2184) 药物原始多热特征向量的嵌入表
    print("对比学习嵌入表的维度: ",embedding_table2.shape[1])

    # 将数据集构成成三元组形式
    # drug_pos_neg = []
    # for drug_id in range(drug_num):
    #     pos = np.where(interaction_matrix[drug_id] == 1)[0]  # 这里可能需要统计一下每一行 1 的个数
    #     neg = np.where(interaction_matrix[drug_id] == 0)[0]
    #     np.random.seed(2024)  # 设置随机种子
    #     for pos_id in pos:
    #         # 为用户的每一个正样本随机匹配一个负样本
    #         neg_id = np.random.choice(neg)
    #         # 将新的三元组加入集合中
    #         drug_pos_neg.append([drug_id, pos_id, neg_id])
    #         # 删除负样本，确保不会被重复选择到
    #         neg = np.delete(neg_id, np.where(neg == neg_id)[0][0])

    return embedding_table1, embedding_table2

def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return sim_matrix


# In[105]:
class DDIDataset(Dataset):
    def __init__(self, list_IDs, df_ddi):
        'Initialization'
        self.list_IDs = list_IDs
        self.df = df_ddi

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        return index

# In[106]:


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return attn, output


# In[107]:

# ATT模块——Multi-head Attention(ATT)
class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)  # Add & Norm 残差连接和层归一化

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        attn_weight, output = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X


# In[108]:


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# In[109]:


# In[112]:
# 这一部分为 encoder-decoder架构
class feature_encoder(torch.nn.Module):  # twin network
    def __init__(self, vector_size, n_heads, n_layers):
        super(feature_encoder, self).__init__()

        self.layers = torch.nn.ModuleList([EncoderLayer(vector_size, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(vector_size)

        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)  # 批量归一化层

        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)

        self.l3 = torch.nn.Linear(vector_size // 4, vector_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(vector_size // 2)

        self.l4 = torch.nn.Linear(vector_size // 2, vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X1 = self.AN(X)
        X2 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X3 = self.l2(X2)

        X4 = self.dr(self.bn3(self.ac(self.l3(self.ac(X3)))))
        X5 = self.l4(X4)

        return X1, X2, X3, X5


# Two Dense Layers架构
class feature_encoder2(torch.nn.Module):  # twin network
    def __init__(self, vector_size):
        super(feature_encoder2, self).__init__()

        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)

        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)
        self.bn2 = torch.nn.BatchNorm1d(vector_size // 4)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.dr(self.bn2(self.ac(self.l2(X))))

        return X
class feature_encoder3(torch.nn.Module):  # twin network 用于对比学习特征降维
    def __init__(self, vector_size, n_heads, n_layers):
        super(feature_encoder3, self).__init__()

        self.layers = torch.nn.ModuleList([EncoderLayer(vector_size, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(vector_size)

        self.l1 = torch.nn.Linear(vector_size, vector_size // 4)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 4)  # 批量归一化层

        self.l2 = torch.nn.Linear(vector_size // 4, vector_size // 8)

        self.l3 = torch.nn.Linear(vector_size // 8, vector_size // 4)
        self.bn3 = torch.nn.BatchNorm1d(vector_size // 4)

        self.l4 = torch.nn.Linear(vector_size // 4, vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X1 = self.AN(X)
        X2 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X3 = self.l2(X2)

        X4 = self.dr(self.bn3(self.ac(self.l3(self.ac(X3)))))
        X5 = self.l4(X4)

        return X3
class Model(torch.nn.Module):
    def __init__(self, input_dim, input_dim1, n_heads, n_layers, event_num=65):
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.input_dim1 = input_dim1
        self.drugEncoder_input_dim = self.input_dim

        self.drugEncoderA = feature_encoder(self.drugEncoder_input_dim, n_heads, n_layers)
        self.drugEncoderB = feature_encoder(self.drugEncoder_input_dim, n_heads, n_layers)

        self.feaEncoder1_3_input_dim = self.drugEncoder_input_dim + self.drugEncoder_input_dim // 4  # 计算DA1+DB3（DA3+DB1）的维度
        self.feaEncoder2_input_dim = self.drugEncoder_input_dim // 2 + self.drugEncoder_input_dim // 2

        self.feaEncoder1 = feature_encoder2(self.feaEncoder1_3_input_dim)
        self.feaEncoder2 = feature_encoder2(self.feaEncoder2_input_dim)
        self.feaEncoder3 = feature_encoder2(self.feaEncoder1_3_input_dim)

        self.feaEncoder1_3_output_dim = self.feaEncoder1_3_input_dim // 4
        self.feaEncoder2_output_dim = self.feaEncoder2_input_dim // 4

        self.feaFui_input_dim = self.feaEncoder1_3_output_dim * 2 + self.feaEncoder2_output_dim + self.drugEncoder_input_dim // 4 * 2

        self.feaFui_contrastive = feature_encoder3(self.input_dim1*2, n_heads, n_layers) # 增强与原始共享编码器，以便进行反向传播

        self.linear_input_dim = self.input_dim1*2 // 8 + self.feaFui_input_dim

        self.l1 = torch.nn.Linear(self.linear_input_dim, (self.linear_input_dim + event_num) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.linear_input_dim + event_num) // 2)

        self.l2 = torch.nn.Linear((self.linear_input_dim + event_num) // 2, event_num)
        self.bn2 = torch.nn.BatchNorm1d(event_num)

        self.ac = gelu

        self.l3 = torch.nn.Linear(event_num, 1)


        self.dr = torch.nn.Dropout(drop_out_rating)

    def forward(self, X, X_origin):
        XA = X[:, 0:self.input_dim]
        XB = X[:, self.input_dim:]

        XA1, XA2, XA3, XAC = self.drugEncoderA(XA)
        XB1, XB2, XB3, XBC = self.drugEncoderB(XB)

        XDC = torch.cat((XAC, XBC), 1)

        X1 = torch.cat((XA1, XB3), 1)
        X2 = torch.cat((XA2, XB2), 1)
        X3 = torch.cat((XA3, XB1), 1)

        X1 = self.feaEncoder1(X1)
        X2 = self.feaEncoder2(X2)
        X3 = self.feaEncoder3(X3)

        XC = self.feaFui_contrastive(X_origin)
        XC1 = self.feaFui_contrastive(torch.stack([perturb_zeros_to_ones(fv.clone(), 0.1) for fv in X_origin]))
        XC2 = self.feaFui_contrastive(torch.stack([shuffle_random_segment(fv.clone(), 0.1) for fv in X_origin]))

        # XC = torch.cat((X1, X2, X3, XA3, XB3), 1)
        # _, _, XC, _ = self.feaFui(XC)  # 这里为什么用的是encoder-decoder架构呢？

        X = torch.cat((XA3, XB3, X1, X2, X3, XC), 1)

        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.dr(self.bn2(self.ac(self.l2(X))))

        X = self.dr(F.relu(self.l3(X)))

        # 返回的这几个分别是最终输出、对比学习输出、特征重构输出
        return X, XC1, XC2, XDC

class Motiv_Fusion(nn.Module):
    def __init__(self, emb_dim1, emb_dim2, bert_n_heads, bert_n_layers, motiv_num=2):
        super(Motiv_Fusion, self).__init__()
        self.sub_motivs = nn.ModuleList([Model(emb_dim1, emb_dim2, bert_n_heads, bert_n_layers) for _ in range(motiv_num)])
        self.attention = MultiHeadAttention(motiv_num, motiv_num, 1)

    def forward(self, x, x_origin, casestudy=False):
        # 每个 sub_motive 的输出为 X, XC1, XC2, XDC
        outputs, XC1_list, XC2_list, XDC_list = [], [], [], []

        for i, sub_motiv in enumerate(self.sub_motivs):
            X, XC1, XC2, pos_XDC = sub_motiv(x, x_origin)  # 解包子模块的多个输出
            outputs.append(X)
            XC1_list.append(XC1)
            XC2_list.append(XC2)
            XDC_list.append(pos_XDC)

        X_cat = torch.cat(outputs, dim=1) # 沿第 0 维拼接 X 的所有输出
        XC1_cat = torch.cat(XC1_list, dim=0) # 沿第 0 维拼接 X 的所有输出
        XC2_cat = torch.cat(XC2_list, dim=0)
        # 将列表转换为 3D 张量
        XDC_tensor = torch.stack(XDC_list, dim=0)# 形状为 (num_submodules, batch_size, input_dim)
        # 在第 0 维进行平均，得到形状为 (batch_size, input_dim) 的二维张量
        XDC_avg = XDC_tensor.mean(dim=0)
        attention_score, scores = self.attention(X_cat)
        """这里的注意力分数（2，batch,batch）相当于注意力网络，是样本对样本的注意力，因此我们把每个头作为特征，抽取其中的一行，对head维度做softmax"""

        scores = torch.sigmoid(scores) # 这里的self.attention(x)可以输出注意力分数
        if casestudy:
            print("X_cat:", X_cat.size())
            return attention_score, scores, outputs, XC1_cat, XC2_cat, XDC_avg
        return scores, outputs, XC1_cat, XC2_cat, XDC_avg

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weight = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        attention_score = torch.matmul(x, self.attention_weight)
        attention_distribution = torch.softmax(attention_score, dim=1)
        return attention_distribution
# 暂时没有用这个对比损失
class con_loss(nn.Module):
    def __init__(self, T=0.05):
        super(con_loss, self).__init__()

        self.T = T

    def forward(self, representations, label):
        n = label.shape[0]
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask) - mask

        mask_dui_jiao_0 = (torch.ones(n, n) - torch.eye(n, n)).to(device)

        similarity_matrix = torch.exp(similarity_matrix / self.T)

        similarity_matrix = similarity_matrix * mask_dui_jiao_0

        sim = mask * similarity_matrix

        no_sim = similarity_matrix - sim

        no_sim_sum = torch.sum(no_sim, dim=1)

        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend
        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(n, n).to(device)  # 防止对角线和不属于同一类的矩阵元素为0，不能取对数
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)

        return loss
class ConLoss(nn.Module):
    def __init__(self, device, temperature=0.5):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, aug_hidden1, aug_hidden2):
        batch_size = aug_hidden1.size(0)
        # 计算相似度 (余弦相似度)
        all_hidden = torch.cat([aug_hidden1, aug_hidden2], dim=0)  # [2*batch_size, projection_dim]
        similarity_matrix = F.cosine_similarity(all_hidden.unsqueeze(1), all_hidden.unsqueeze(0), dim=2)

        # 创建掩码，防止计算自身的相似度
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)

        # 将自身的相似度置为负无穷
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # 计算 Contrastive Loss, 正例是 (i, i+batch_size)，即同一个样本的两个增强版本
        positive_sim = torch.cat(
            [torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)],
            dim=0)
        loss = -torch.log(torch.exp(positive_sim / self.temperature) / torch.exp(similarity_matrix / self.temperature).sum(dim=1))
        # 总共2*batch_size个正例对，取他们损失的平均值，想加取平均值里面内涵了极大似然估计

        return loss.mean()
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, scores_pos, scores_neg):
        loss = -F.logsigmoid(scores_pos - scores_neg)
        return torch.mean(loss)  # 返回损失的平均值
class my_loss(nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

        self.criteria1 = BPRLoss()
        self.criteria2 = torch.nn.MSELoss()
        self.criteria3 = ConLoss(device,temperature=con_loss_T)

    def forward(self, scores_pos, scores_neg, aug_hidden1, aug_hidden2, motivation, XDC, inputs):
        # 对比学习损失的系数可能要大一些，比如100？
        loss = self.criteria1(scores_pos, scores_neg) + \
               10*self.criteria2(inputs.float(), XDC) + \
               0.1 * self.criteria3(aug_hidden1, aug_hidden2) + \
               0.1 * mutual_information_loss(motivation)

        return loss


def BERT_train(training_generator, Testing_generator, embeddingtable1, embeddingtable2):
    motiv_model = Motiv_Fusion(len(embeddingtable1[0]), len(embeddingtable2[0]), bert_n_heads, bert_n_layers, motiv_num=motiv_num)
    # motiv_model = torch.nn.DataParallel(motiv_model)
    motiv_model = motiv_model.cuda()
    model_optimizer = RAdam(motiv_model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate)
    # 记录训练误差和测试误差
    Loss = []
    # 早期停止策略，阈值设置为 7
    patience=0
    min_test_loss=100.0
    headers = ["running_loss", "testing_loss"]
    with open('./5Casestudy/MMDDI2-loss.csv', "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for epoch in range(epo_num):
            My_loss = my_loss()

            running_loss = 0.0

            motiv_model.train()
            for batch_idx, index in tqdm(enumerate(training_generator, 0), total=len(training_generator), desc="Train",
                                         unit="batch"):
                data = extraction.iloc[index]
                drugA = [drug_to_index[i] for i in data["drugA"].values.tolist()]
                pos = [drug_to_index[j] for j in data["drugB"].values.tolist()]
                neg = []
                """这里要记得考虑匹配的负样本重复问题"""
                for i in drugA:
                    neg_drug = np.random.randint(drug_num)
                    while ((extraction['drugA'] == df_drug.iloc[i].name) & (
                            extraction['drugB'] == df_drug.iloc[neg_drug].name)).any() or (i == neg_drug):
                        neg_drug = np.random.randint(drug_num)
                    neg.append(neg_drug)

                inputs1_pos = np.hstack((np.array(embeddingtable1[drugA]), np.array(embeddingtable1[pos])))
                inputs1_pos = np.vstack((inputs1_pos, np.hstack((np.array(embeddingtable1[pos]), np.array(embeddingtable1[drugA])))))  # BERT双头的本质

                inputs2_pos = np.hstack((np.array(embeddingtable2[drugA]), np.array(embeddingtable2[pos])))
                inputs2_pos = np.vstack((inputs2_pos, np.hstack((np.array(embeddingtable2[pos]), np.array(embeddingtable2[drugA])))))  # BERT双头的本质

                inputs1_neg = np.hstack((np.array(embeddingtable1[drugA]), np.array(embeddingtable1[neg])))
                inputs1_neg = np.vstack((inputs1_neg, np.hstack((np.array(embeddingtable1[neg]), np.array(embeddingtable1[drugA])))))  # BERT双头的本质

                inputs2_neg = np.hstack((np.array(embeddingtable2[drugA]), np.array(embeddingtable2[neg])))
                inputs2_neg = np.vstack((inputs2_neg, np.hstack((np.array(embeddingtable2[neg]), np.array(embeddingtable2[drugA])))))  # BERT双头的本质

                inputs1_pos = torch.from_numpy(inputs1_pos).cuda()
                inputs2_pos = torch.from_numpy(inputs2_pos).cuda()
                inputs1_neg = torch.from_numpy(inputs1_neg).cuda()
                inputs2_neg = torch.from_numpy(inputs2_neg).cuda()

                pos_scores, output1, XC1, XC2, pos_XDC = motiv_model(inputs1_pos.float(), inputs2_pos.float())
                neg_scores, output2, _1, _2, neg_XDC = motiv_model(inputs1_neg.float(), inputs2_neg.float())
                # 将每个模型的输出进行合并
                output = [torch.cat((o1, o2), dim=0) for o1, o2 in zip(output1, output2)]
                # output = torch.cat(output,dim=1)

                # loss = My_loss(pos_scores, neg_scores, np.vstack((pos_X_de,neg_X_de)),XC1, XC2, np.vstack((pos_XDC, neg_XDC)), np.vstack((inputs1_pos, inputs1_neg)))
                loss = My_loss(pos_scores, neg_scores, XC1, XC2, output, torch.cat((pos_XDC, neg_XDC), dim=0), torch.cat((inputs1_pos, inputs1_neg), dim=0))

                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()
                if (batch_idx % 20 == 0):
                    print('Training at Epoch ' + str(epoch + 1) + ' iteration ' + str(batch_idx) + ', total loss is ' + '%.6f' % (loss.item() / (2 * batch_size)))
                running_loss += loss.item()
                writer.writerow([loss.item() / (2 * batch_size), 0])

            motiv_model.eval()
            testing_loss = 0.0
            with torch.no_grad():
                for batch_idx, index in tqdm(enumerate(Testing_generator, 0), total=len(Testing_generator), desc="Test",
                                             unit="batch"):
                    data = extraction.iloc[index]
                    drugA = [drug_to_index[i] for i in data["drugA"].values.tolist()]
                    pos = [drug_to_index[j] for j in data["drugB"].values.tolist()]
                    neg = []
                    for i in drugA:
                        neg_drug = np.random.randint(drug_num)
                        while ((extraction['drugA'] == df_drug.iloc[i].name) & (
                                extraction['drugB'] == df_drug.iloc[neg_drug].name)).any() or (i == neg_drug):
                            neg_drug = np.random.randint(drug_num)
                        neg.append(neg_drug)

                    inputs1_pos = np.hstack((np.array(embeddingtable1[drugA]), np.array(embeddingtable1[pos])))
                    inputs1_pos = np.vstack((inputs1_pos, np.hstack(
                        (np.array(embeddingtable1[pos]), np.array(embeddingtable1[drugA])))))  # BERT双头的本质

                    inputs2_pos = np.hstack((np.array(embeddingtable2[drugA]), np.array(embeddingtable2[pos])))
                    inputs2_pos = np.vstack((inputs2_pos, np.hstack(
                        (np.array(embeddingtable2[pos]), np.array(embeddingtable2[drugA])))))  # BERT双头的本质

                    inputs1_neg = np.hstack((np.array(embeddingtable1[drugA]), np.array(embeddingtable1[neg])))
                    inputs1_neg = np.vstack((inputs1_neg, np.hstack(
                        (np.array(embeddingtable1[neg]), np.array(embeddingtable1[drugA])))))  # BERT双头的本质

                    inputs2_neg = np.hstack((np.array(embeddingtable2[drugA]), np.array(embeddingtable2[neg])))
                    inputs2_neg = np.vstack((inputs2_neg, np.hstack(
                        (np.array(embeddingtable2[neg]), np.array(embeddingtable2[drugA])))))  # BERT双头的本质

                    inputs1_pos = torch.from_numpy(inputs1_pos).cuda()
                    inputs2_pos = torch.from_numpy(inputs2_pos).cuda()
                    inputs1_neg = torch.from_numpy(inputs1_neg).cuda()
                    inputs2_neg = torch.from_numpy(inputs2_neg).cuda()

                    pos_scores, output1, XC1, XC2, pos_XDC = motiv_model(inputs1_pos.float(), inputs2_pos.float())
                    neg_scores, output2, _1, _2, neg_XDC = motiv_model(inputs1_neg.float(), inputs2_neg.float())
                    # 将每个模型的输出进行合并
                    output = [torch.cat((o1, o2), dim=0) for o1, o2 in zip(output1, output2)]

                    loss = My_loss(pos_scores, neg_scores, XC1, XC2, output, torch.cat((pos_XDC, neg_XDC), dim=0),
                                   torch.cat((inputs1_pos, inputs1_neg), dim=0))
                    if (batch_idx % 20 == 0):
                        print('Test at Epoch ' + str(epoch + 1) + ' iteration ' + str(
                            batch_idx) + ', testing loss is ' + '%.6f' % (loss.cpu().detach().numpy() / batch_size))
                    testing_loss += loss.item()
                    writer.writerow([0, loss.item() / batch_size])
            print('epoch [%d] training loss: %.6f testing_loss: %.6f ' % (
            epoch + 1, running_loss / (len(training_generator.dataset) * 2),
            testing_loss / len(Testing_generator.dataset)))
            Loss.append(
                [running_loss / (len(training_generator.dataset) * 2), testing_loss / len(Testing_generator.dataset)])
            writer.writerow(Loss[-1])
            if Loss[-1][-1] < min_test_loss and Loss[-1][-1]/min_test_loss<0.98:
                patience = 0
                min_test_loss = Loss[-1][-1]
            else:
                patience += 1
                if patience>=7:
                    break
    # # 保存模型
    # path = "./data/MMDDI2"
    # filename = f"{path}_epoch{epo_num}.pt"
    # torch.save(motiv_model.state_dict(), filename)
    # print(f"Model saved as {filename}")
    # # 加载模型
    # # motiv_model.load_state_dict(torch.load(path))
    return RecommendTest(Testing_generator, motiv_model)

def RecommendTest(data_generator, recommender):
    recommender.eval()
    hits_ratio = 0.0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    drugA_num = 0.0
    targets = []
    pre_scores = np.zeros((0, topk), dtype=float)
    recommend_drugs = np.zeros((0, topk), dtype=int)
    with torch.no_grad():
        for batch_idx, index in enumerate(data_generator, 0):
            hits = 0.0
            ndcg = 0.0
            mrr = 0.0
            test_data = extraction.iloc[index]
            drugA_num += len(test_data['drugA'].unique())
            for drugA_name in test_data['drugA'].unique():
                drugA_data = test_data[test_data['drugA'] == drugA_name]
                pos = np.array([drug_to_index[i] for i in drugA_data['drugB'].values.tolist()])
                # pos_all = np.where(interaction_matrix[drugA_id]==1)

                # 生成候选集，采取留一法leave-one-out来进行配对嘛？
                all_drugs = set(range(drug_num))
                num_samples = int(topk*1.5)
                neg_candidate = random.sample(list(all_drugs - set(pos)), num_samples)
                pos_candidate = [np.random.choice(pos)]
                """这里要把候选项打乱"""
                candidate_drugs = pos_candidate + neg_candidate

                # 计算推荐分数
                drugA_tensor = torch.tensor([drug_to_index[drugA_name]] * len(candidate_drugs), dtype=torch.long)
                candidate_tensor = torch.tensor(candidate_drugs, dtype=torch.long)

                inputs1 = np.hstack((np.array(embeddingtable1[drugA_tensor.tolist()]), np.array(embeddingtable1[candidate_drugs])))
                inputs2 = np.hstack((np.array(embeddingtable2[drugA_tensor.tolist()]), np.array(embeddingtable2[candidate_drugs])))

                inputs1 = torch.from_numpy(inputs1).cuda()
                inputs2 = torch.from_numpy(inputs2).cuda()

                pre_score, _, _, _, _ = recommender(inputs1.float(), inputs2.float())
                pre_score = pre_score.sum(dim=1).cpu()
                # 排序得到风险级别较高的药物
                recommend_drugs_id = candidate_tensor[pre_score.argsort(descending=True)[:topk]]
                # pre_score = pre_score.detach().cpu().numpy()
                pre_scores = np.vstack((pre_scores, pre_score[pre_score.argsort(descending=True)][:topk].reshape(-1, topk)))
                recommend_drugs = np.vstack((recommend_drugs, recommend_drugs_id.reshape(-1, topk)))
                targets = targets + pos_candidate
                # 计算指标
                for rank, drug in enumerate(recommend_drugs_id):
                    # 通过改变in后面的范围，可以分别产生三个不同的场景
                    if drug == pos_candidate[0]:
                        hits += 1
                        ndcg += 1 / np.log2(rank + 2)  # NDCG
                        mrr += 1 / (rank + 1)  # MRR
                        break  # 只计算第一个命中
            if (batch_idx % 20 == 0):
                print(f'batch{batch_idx} --hits_ratio: {hits/len(test_data["drugA"].unique()):.4f}, NDCG: {ndcg/len(test_data["drugA"].unique()):.4f}, MRR: {mrr/len(test_data["drugA"].unique()):.4f}')
            hits_ratio += hits
            ndcg_sum += ndcg
            mrr_sum += mrr

    print(f'Final validation--hits_ratio: {hits_ratio/drugA_num:.6f}, NDCG: {ndcg_sum/drugA_num:.6f}, MRR: {mrr_sum/drugA_num:.6f}')
    pre_scores = torch.from_numpy(pre_scores)
    recommend_drugs = torch.from_numpy(recommend_drugs)
    targets = torch.tensor(targets)
    # print('pre_scores:', pre_scores)
    # print('recommend_drugs:', recommend_drugs)
    # print('targets:', targets)
    if args.c:
        case_analysis(['Dextroamphetamine', 'Fenfluramine', 'Bortezomib', 'Clarithromycin'], recommender)
    return recommend_drugs, targets, drugA_num

def case_analysis(anchor_drug_list, model):
    """
    针对特定锚点药物的案例分析
    Args:
        anchor_drug: 锚点药物ID (如'DB01576')
        model: 训练好的预测模型
        extraction_df: DDI提取表
        embedding_table1/2: 两种特征嵌入表
        drug_to_index: 药物ID到索引的映射
        index_to_drug: 索引到药物ID的映射
        topk: 返回前K个高风险药物
    Returns:
        pd.DataFrame 包含药物ID、风险得分、注意力分数的结果
    """
    for anchor_drug in anchor_drug_list:
        # 获取正例药物
        pos_drugs = extraction[extraction['drugA'] == anchor_drug]['drugB'].unique()
        pos_indices = [drug_to_index[d] for d in pos_drugs if d in drug_to_index]
        
        # 生成候选集（正例+负例）
        all_drugs = set(range(len(drug_to_index)))
        # neg_candidate = random.sample(list(all_drugs - set(pos_indices)),20)
        neg_candidate = list(all_drugs - set(pos_indices))
        candidates = pos_indices + neg_candidate
        
        # 构建输入数据
        anchor_idx = drug_to_index[anchor_drug]
        anchor_emb1 = np.repeat([embeddingtable1[anchor_idx]], len(candidates), axis=0)
        candidate_emb1 = embeddingtable1[candidates]
        inputs1 = np.hstack((anchor_emb1, candidate_emb1))
        
        anchor_emb2 = np.repeat([embeddingtable2[anchor_idx]], len(candidates), axis=0)
        candidate_emb2 = embeddingtable2[candidates]
        inputs2 = np.hstack((anchor_emb2, candidate_emb2))
        
        # 模型预测
        with torch.no_grad():
            inputs1 = torch.FloatTensor(inputs1).cuda()
            inputs2 = torch.FloatTensor(inputs2).cuda()
            attn, pred_scores, _, _, _, _ = model(inputs1, inputs2, True)  # 假设模型返回注意力
            
        # 解析注意力分数
        # avg_attn = (attn1.mean(dim=1) + attn2.mean(dim=1)).cpu().numpy()  # 综合两种模态注意力
        
        # 生成结果DataFrame
        print('pred_scores:', pred_scores.sum(dim=1).size())
        print('attn:', attn.size())
        print('Attention_head1:', attn[:, 0].size())
        print('Attention_head1_mean:', attn[:, 0].mean(dim=1).size())
        print(attn[:, 0])
        results = pd.DataFrame({
            'DrugID': [df_drug.iloc[i].id for i in candidates],
            'RiskScore': pred_scores.sum(dim=1).cpu().numpy(),
            'Attention_head1': attn[:, 0].mean(dim=1).cpu().numpy(),
            'Attention_head2': attn[:, 1].mean(dim=1).cpu().numpy(),
        })
        
        # 按风险分排序并添加排名
        results = results.sort_values('RiskScore', ascending=False).head(20)
        results['Rank'] = range(1, len(results)+1)
        out_results = results[['Rank', 'DrugID', 'RiskScore', 'Attention_head1', 'Attention_head2']]
        out_results.to_csv(f"./5Casestudy/{anchor_drug}_top20.csv")
        print(f"{anchor_drug} ({df_drug.iloc[df_drug['name']==anchor_drug].id}) 分析结果:")
        print(out_results)
    return 


# In[116]:
def cross_val(embeddingtable1, embeddingtable2):
    kf = KFold(n_splits=cross_ver_tim, shuffle=True, random_state=2024)
    headers = [f"HR@{topk}", f"NDCG@{topk}", f"MRR@{topk}", f"Recall@{topk}"]
    with open('./5Casestudy/MMDDI2-all' + '.csv', "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for train_index, test_index in kf.split(extraction):
            partition_sup = {'train': train_index, 'Test': test_index}
            training_set = DDIDataset(partition_sup['train'], extraction)
            training_generator = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

            Testing_set = DDIDataset(partition_sup['Test'], extraction)
            Testing_generator = DataLoader(dataset=Testing_set, batch_size=batch_size, shuffle=False)
            print("train len = ", training_set.__len__(), "\t", "arg train len = ", 2 * training_set.__len__(), "\t",
                  "test len = ", Testing_set.__len__())

            recommendation, target, drugA_num = BERT_train(training_generator, Testing_generator, embeddingtable1, embeddingtable2)

            hit_k = Metric.HIT(recommendation, target, topk) / drugA_num
            ndcg_k = Metric.NDCG(recommendation, target, topk) / drugA_num
            mrr_k = Metric.MRR(recommendation, target, topk) / drugA_num
            recall_k = Metric.RECALL(recommendation, target, topk) / drugA_num
            writer.writerow([hit_k, ndcg_k, mrr_k, recall_k])
            break 

# In[119]:
if __name__ == '__main__':
    embeddingtable1, embeddingtable2 = prepare(feature_list)
    start = time.time()
    cross_val(embeddingtable1, embeddingtable2)
    print("time used:", (time.time() - start) / 3600)



