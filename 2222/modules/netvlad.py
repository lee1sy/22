class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size=128, cluster_size=64, output_dim=256, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.output_dim = output_dim
        
        # 聚类中心权重
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * (1 / math.sqrt(feature_size)))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * (1 / math.sqrt(feature_size)))
        
        # 降维层：将巨大的 8192 维压缩回 256 维
        self.hidden1_weights = nn.Parameter(torch.randn(cluster_size * feature_size, output_dim) * (1 / math.sqrt(feature_size)))
        
        # 用 LayerNorm 代替 BatchNorm，完美适配 batch_size=1
        self.norm1 = nn.LayerNorm(cluster_size)
        
        self.gating = GatingContext(output_dim) if gating else None

    def forward(self, x, mask=None):
        B, N, D = x.shape
        # 1. 计算软分配权重
        activation = torch.matmul(x, self.cluster_weights) # [B, N, C]
        activation = self.norm1(activation)
        activation = F.softmax(activation, dim=-1)

        # 2. 屏蔽无效点 (非常关键！)
        if mask is not None:
            activation = activation * mask.unsqueeze(-1).to(x.dtype)
        
        # 3. 计算残差聚合
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2
        
        vlad = torch.matmul(activation.transpose(1, 2), x) # [B, C, D]
        vlad = vlad.transpose(1, 2) - a
        
        # 4. 归一化与降维
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape(B, -1)
        vlad = F.normalize(vlad, dim=1, p=2)
        
        vlad = torch.matmul(vlad, self.hidden1_weights)
        
        if self.gating:
            vlad = self.gating(vlad)
            
        return F.normalize(vlad, p=2, dim=1)

class GatingContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * (1 / math.sqrt(dim)))
        self.norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sigmoid(self.norm(torch.matmul(x, self.gating_weights)))
        return x * gates