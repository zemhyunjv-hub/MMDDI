import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def perturb_zeros_to_ones(feature_vector, ratio=0.1):
    """
    扰动一定比例的0变为1
    :param feature_vector: 输入的特征向量，1D PyTorch tensor
    :param ratio: 扰动的比例，默认为10%
    :return: 扰动后的特征向量
    """
    # 获取0的位置
    zero_indices = torch.where(feature_vector == 0)[0]
    # 需要变成1的0的数量
    num_to_flip = int(len(zero_indices) * ratio)
    # 随机选择要翻转的0的位置
    flip_indices = zero_indices[torch.randperm(len(zero_indices))[:num_to_flip]]
    # 将这些位置的0变为1
    feature_vector[flip_indices] = 1
    return feature_vector


def shuffle_random_segment(feature_vector, ratio=0.1):
    """
    随机选择一个连续的片段进行shuffle，保证1的数量不变
    :param feature_vector: 输入的特征向量，1D PyTorch tensor
    :param ratio: 选择片段的比例，默认为10%
    :return: 扰动后的特征向量
    """
    length = len(feature_vector) # 确定片段的长度
    segment_length = int(length * ratio)
    # 随机选择片段的起始位置
    start_idx = torch.randint(0, length - segment_length + 1, (1,)).item()
    end_idx = start_idx + segment_length
    # 对这个片段进行shuffle
    segment = feature_vector[start_idx:end_idx]
    shuffled_segment = segment[torch.randperm(len(segment))]
    # 替换原特征向量中的片段
    feature_vector[start_idx:end_idx] = shuffled_segment

    return feature_vector


# 示例
# if __name__ == "__main__":
#     # 创建一个随机的0和1组成的特征向量 (torch.Tensor)
#     feature_vector = torch.tensor(torch.bernoulli(torch.full((100,), 0.3)).long())
#     print("原始特征向量：", feature_vector)
#
#     # 应用第一种扰动策略
#     perturbed_vector_1 = perturb_zeros_to_ones(feature_vector.clone(), ratio=0.1)
#     print("扰动后的特征向量（0变1）：", perturbed_vector_1)
#
#     # 应用第二种扰动策略
#     perturbed_vector_2 = shuffle_random_segment(feature_vector.clone(), ratio=0.1)
#     print("扰动后的特征向量（片段shuffle）：", perturbed_vector_2)


class ContrastiveLearningModel(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(ContrastiveLearningModel, self).__init__()
        # 一个简单的前馈网络作为投影头，将特征向量映射到低维潜在空间
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, projection_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)  # 在投影空间中进行L2归一化

class InfoNCELoss(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss, another implementation.
    """

    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.tem = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden1, aug_hidden2):
        """
        Args:
            aug_hidden1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden1.ndim > 2:
            # flatten tensor
            aug_hidden1 = aug_hidden1.view(aug_hidden1.size(0), -1)
            aug_hidden2 = aug_hidden2.view(aug_hidden2.size(0), -1)

        current_batch = aug_hidden1.size(0)
        N = 2 * current_batch
        all_hidden = torch.cat((aug_hidden1, aug_hidden2), dim=0)  # [2*B, D]

        sim = F.cosine_similarity(all_hidden.unsqueeze(1), all_hidden.unsqueeze(0), dim=2) / self.tem

        sim_i_j = torch.diag(sim, current_batch)
        sim_j_i = torch.diag(sim, -current_batch)
        mask = torch.ones((N, N)).bool()
        mask = mask.fill_diagonal_(0)
        index1 = torch.arange(batch_size) + batch_size
        index2 = torch.arange(batch_size)
        index = torch.cat([index1, index2], 0).unsqueeze(-1)  # [2*B, 1]
        mask = torch.scatter(mask, -1, index, 0) # 沿着最后一维根据index指定的位置将值0散步到mask矩阵中
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        nce_loss = self.criterion(logits, labels)

        return nce_loss

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


# 示例：使用对比学习
# if __name__ == "__main__":
#     # 定义超参数
#     input_dim = 16  # 输入的特征向量长度
#     projection_dim = 8  # 投影空间的维度
#     batch_size = 4
#     learning_rate = 1e-3
#     epochs = 1
#
#     # 初始化模型和优化器
#     model = ContrastiveLearningModel(input_dim=input_dim, projection_dim=projection_dim)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     # 假设我们有一个批次的原始特征向量
#     for epoch in range(epochs):
#         feature_vector_batch = torch.bernoulli(torch.full((batch_size, input_dim), 0.3)).long()
#
#         # 使用数据增强生成两组增强后的视图
#         augmented_batch_1 = torch.stack([perturb_zeros_to_ones(fv.clone(), 0.1) for fv in feature_vector_batch])
#         augmented_batch_2 = torch.stack([shuffle_random_segment(fv.clone(), 0.1) for fv in feature_vector_batch])
#
#         # 将增强后的视图通过投影头
#         z_i = model(augmented_batch_1.float())
#         z_j = model(augmented_batch_2.float())
#
#         # 计算对比学习损失
#         loss = ConLoss(z_i, z_j)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
