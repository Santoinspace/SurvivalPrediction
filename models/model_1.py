"""
模型架构
- 3D CNN 模块：用于从 3D PET/CT 数据中提取特征。
- 表格数据编码器：用于将 21 维的临床数据映射到 512 维度。
- 多模态融合模块：将 3D CNN 提取的图像特征和表格数据编码器的特征进行融合。
- 预测模块：使用融合后的特征进行生存分析预测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# 1. 基础模块定义
# -------------------------------

class Simple3DCNN(nn.Module):
    """
    简单 3D CNN 模块，用于从 3D PET/CT 数据中提取特征。
    输入尺寸：(batch_size, 1, 112, 112, 112)
    输出尺寸：(batch_size, feature_dim) 其中 feature_dim=512
    """
    def __init__(self, in_channels=1, feature_dim=512):
        super(Simple3DCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),  # 112 -> 56
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),  # 56 -> 28
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),  # 28 -> 14
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)   # 14 -> 7
        )
        # 经过卷积后尺寸为: (batch, 256, 7, 7, 7)
        self.fc = nn.Linear(256 * 7 * 7 * 7, feature_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x  # (batch_size, feature_dim)


class TabularEncoder(nn.Module):
    """
    表格数据编码器，用于将 21 维的临床数据映射到 feature_dim（512）维度。
    """
    def __init__(self, input_dim=21, feature_dim=512):
        super(TabularEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
    
    def forward(self, x):
        return self.fc(x)  # (batch_size, feature_dim)


# -------------------------------
# 2. 多模态融合模型定义
# -------------------------------

class MultiModalFusionModel(nn.Module):
    """
    多模态融合模型：
      - PET 和 CT 数据均为 3D 图像 (batch_size, 1, 112, 112, 112)
      - 表格数据为 (batch_size, 21)
      
    模型流程：
      1. 分别使用 3D CNN 提取 PET 和 CT 特征，并融合生成图像特征 x (维度 512)
      2. 使用 TabularEncoder 提取表格数据特征 y (维度 512)
      3. 对 x 和 y 分别使用 Transformer 模块进行模态内特征交互（intra）
      4. 利用交叉注意力模块 (cross attention) 计算图像与表格特征之间的互补信息
      5. 将图像特征、交叉注意力输出和表格特征进行拼接融合，生成最终特征，再经过全连接层输出 interval_num 维结果（生存概率）
    """
    def __init__(self, pet_in_channels=1, ct_in_channels=1, tabular_dim=21,
                 feature_dim=512, num_heads=8, transformer_layers=1, interval_num=4):
        super(MultiModalFusionModel, self).__init__()
        
        # 图像分支：分别提取 PET 和 CT 特征
        self.pet_encoder = Simple3DCNN(in_channels=pet_in_channels, feature_dim=feature_dim)
        self.ct_encoder  = Simple3DCNN(in_channels=ct_in_channels, feature_dim=feature_dim)
        
        # 使用 Transformer 编码器对图像特征进行模态内信息交互
        encoder_layer_img = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, batch_first=True)
        self.transformer_intra_img = nn.TransformerEncoder(encoder_layer_img, num_layers=transformer_layers)
        
        # 表格数据编码器
        self.tabular_encoder = TabularEncoder(input_dim=tabular_dim, feature_dim=feature_dim)
        encoder_layer_tab = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, batch_first=True)
        self.transformer_intra_tab = nn.TransformerEncoder(encoder_layer_tab, num_layers=transformer_layers)
        
        # 交叉注意力模块：使用图像特征作为 Query，表格特征作为 Key 和 Value
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        
        # 融合层：拼接图像、交叉注意力输出和表格特征后融合
        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, interval_num),
            nn.Sigmoid()  # 输出概率，范围在 [0,1] 内
        )
    
    def forward(self, pet, ct, tabular):
        """
        参数：
          pet: PET 3D 图像张量，形状 (B, 1, 112, 112, 112)
          ct: CT 3D 图像张量，形状 (B, 1, 112, 112, 112)
          tabular: 表格数据，形状 (B, 21)
        """
        # 1. 图像特征提取
        pet_feat = self.pet_encoder(pet)  # (B, 512)
        ct_feat  = self.ct_encoder(ct)     # (B, 512)
        # 简单融合：这里使用均值融合 PET 与 CT 特征
        image_feat = (pet_feat + ct_feat) / 2  # (B, 512)
        
        # 为 Transformer 添加序列维度：转换为 (B, 1, 512)
        image_feat_seq = image_feat.unsqueeze(1)
        # 模态内（intra）Transformer 处理图像特征
        image_feat_trans = self.transformer_intra_img(image_feat_seq)  # (B, 1, 512)
        
        # 2. 表格数据特征提取
        tab_feat = self.tabular_encoder(tabular)  # (B, 512)
        tab_feat_seq = tab_feat.unsqueeze(1)        # (B, 1, 512)
        tab_feat_trans = self.transformer_intra_tab(tab_feat_seq)  # (B, 1, 512)
        
        # 3. 交叉注意力：图像特征作为 Query，与表格特征进行交互
        cross_attn_output, _ = self.cross_attention(query=image_feat_trans,
                                                    key=tab_feat_trans,
                                                    value=tab_feat_trans)  # (B, 1, 512)
        
        # 4. 融合特征：拼接图像特征、交叉注意力输出和表格特征（去除序列维度）
        fused = torch.cat([image_feat_trans.squeeze(1),
                           cross_attn_output.squeeze(1),
                           tab_feat_trans.squeeze(1)], dim=-1)  # (B, 512*3)
        
        # 5. 预测输出：最终输出 interval_num 个生存概率
        output = self.fusion_fc(fused)  # (B, interval_num)
        return output


# -------------------------------
# 3. 示例代码：如何使用该模型
# -------------------------------
if __name__ == "__main__":
    # 假设批量大小为 2（仅为示例）
    batch_size = 2
    interval_num = 4  # 输出 4 个时间区间的生存概率
    
    # 构造伪造数据：
    # PET 与 CT 3D 图像数据，尺寸 (B, 1, 112, 112, 112)
    pet  = torch.randn(batch_size, 1, 112, 112, 112)
    ct   = torch.randn(batch_size, 1, 112, 112, 112)
    # 表格数据，尺寸 (B, 21)
    tabular = torch.randn(batch_size, 21)
    
    # 构造模型实例
    model = MultiModalFusionModel(pet_in_channels=1, ct_in_channels=1, tabular_dim=21,
                                  feature_dim=512, num_heads=8, transformer_layers=1,
                                  interval_num=interval_num)
    
    # 前向传播，得到预测结果
    output = model(pet, ct, tabular)
    print("预测输出形状:", output.shape)  # 预期输出形状: (batch_size, interval_num)
    print("预测输出:", output)

    # 统计模型参数量
    model_name = 'MultiModalFusionModel'
    total_comsuption = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(f'model name:{model_name}  Number of parameter: {(total_comsuption / 1024 / 1024):.4f}M')
