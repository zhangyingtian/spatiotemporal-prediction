from 时空图论文.结果.置信区间训练 import calculate_standard_score
from 自己设计的.训练.数据处理 import prepare_data, process_and_save_data, cluster, oc_history_cols
from 自己设计的.训练.训练0 import  Train_Module, score_func, evaluate_finaltest_rul, RMSELoss
from 自己设计的.训练.plot import plot_sensors, train_predicted, fianl_test_predicted, plot_engine_comparison_combined, \
    plot_engine_comparison_combined_2, plot_error_distribution_from_csv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd


# 训练 Loss: 13.3105 | 评估 Loss: 11.7452
# 编码模块
class PositionalEncoding00(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


# 注意力池化模块（时序部分）
class AttentionPooling(nn.Module):
    """注意力池化层：自适应加权聚合时序特征"""

    def __init__(self, embed_dim):
        super().__init__()
        # 注意力机制计算权重
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)  # 时间维度上归一化
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数：避免零梯度"""
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 1e-4)  # 小偏置避免零权重

    def forward(self, x):
        """
        输入:
            x - 时序特征 [B, T, D] (批次大小, 时间步数, 特征维度)
        输出:
            pooled - 池化特征 [B, D]
            weights - 注意力权重 [B, T] (用于可视化分析)
        """
        # 计算注意力分数 [B, T, 1]
        attn_scores = self.attention(x)

        # 计算加权特征 [B, T, D] * [B, T, 1] -> [B, D]
        weighted_features = torch.sum(x * attn_scores, dim=1)

        return weighted_features, attn_scores.squeeze(-1)


class Conv1DDynamicGraphSpatialExtractor02(nn.Module):
    """使用Conv1D替代GRU的动态图空间特征提取器，输出[B, D]"""

    def __init__(self, num_nodes, conv_hidden=8, k_cheb=2, spatial_dim=32, conv_kernel=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.k_cheb = k_cheb
        self.spatial_dim = spatial_dim

        # Conv1D时序特征提取
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=conv_hidden,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2
            ),
            nn.ReLU(inplace=True)
        )
        self.conv_hidden = conv_hidden

        # 残差连接
        self.residual_proj = nn.Conv1d(
            in_channels=1,
            out_channels=conv_hidden,
            kernel_size=1
        )

        # 动态图参数
        self.weight_key = nn.Parameter(torch.empty(conv_hidden, 1))
        self.weight_query = nn.Parameter(torch.empty(conv_hidden, 1))

        # 图卷积参数
        self.cheb_weights = nn.Parameter(torch.empty(k_cheb, spatial_dim))
        self.output_proj = nn.Linear(spatial_dim, spatial_dim)

        # 特征融合 & 全局池化
        self.fusion_layer = nn.Linear(spatial_dim + conv_hidden, spatial_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局池化层

        # 参数初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        # 卷积层参数
        nn.init.kaiming_normal_(self.conv1d[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1d[0].bias)

        # 残差连接
        nn.init.kaiming_normal_(self.residual_proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.residual_proj.bias)

        # 动态图参数
        nn.init.xavier_uniform_(self.weight_key.data)
        nn.init.xavier_uniform_(self.weight_query.data)

        # 图卷积参数
        nn.init.xavier_uniform_(self.cheb_weights.data)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.zeros_(self.output_proj.bias.data)

        # 特征融合层
        nn.init.xavier_uniform_(self.fusion_layer.weight.data)
        nn.init.zeros_(self.fusion_layer.bias.data)

        # # 时间聚合层初始化
        # nn.init.kaiming_uniform_(self.temporal_aggregate[0].weight, nonlinearity='relu')
        # nn.init.zeros_(self.temporal_aggregate[0].bias)
        # nn.init.xavier_uniform_(self.temporal_aggregate[3].weight)
        # nn.init.zeros_(self.temporal_aggregate[3].bias)

    def build_dynamic_graph(self, node_rep):
        """动态图构建"""
        B, N, D = node_rep.shape

        # 计算自注意力分数
        key = torch.matmul(node_rep, self.weight_key)  # [B, N, 1]
        query = torch.matmul(node_rep, self.weight_query)  # [B, N, 1]

        # 对称注意力矩阵
        attn_scores = key + query.permute(0, 2, 1)  # [B, N, N]
        attn_scores = F.leaky_relu(attn_scores, negative_slope=0.01)

        # 对称归一化
        adj = F.softmax(attn_scores, dim=-1)
        adj = 0.5 * (adj + adj.permute(0, 2, 1))  # 强制对称

        # 计算归一化拉普拉斯矩阵
        deg = torch.sum(adj, dim=-1, keepdim=True)  # [B, N, 1]
        deg_inv_sqrt = torch.where(
            deg > 0,
            deg.pow(-0.5),
            torch.zeros_like(deg)
        )

        # L = I - D^{-1/2} A D^{-1/2}
        norm_adj = deg_inv_sqrt * adj * deg_inv_sqrt.permute(0, 2, 1)
        return torch.eye(N, device=node_rep.device).unsqueeze(0) - norm_adj

    def chebyshev_polynomials(self, laplacian):
        """切比雪夫多项式基"""
        B, N, _ = laplacian.shape
        supports = torch.zeros(B, self.k_cheb, N, N, device=laplacian.device)

        # 使用高效的多项式计算
        if self.k_cheb > 0:
            supports[:, 0] = torch.eye(N, device=laplacian.device)

        if self.k_cheb > 1:
            supports[:, 1] = laplacian

            # 迭代计算高阶多项式
            for k in range(2, self.k_cheb):
                # 使用原地操作减少内存分配
                supports[:, k] = 2 * torch.matmul(laplacian, supports[:, k - 1])
                supports[:, k].sub_(supports[:, k - 2])

        return supports

    def graph_convolution(self, x, supports):
        """图卷积操作"""
        # 多阶信号聚合: [B, K, N, T]
        agg_features = torch.einsum('bkmn,bnt->bknt', supports, x)

        # 线性组合多阶特征
        output = torch.einsum(
            'kf,bknt->bfnt',
            self.cheb_weights,
            agg_features
        )

        # 节点维度池化: [B, spatial_dim, T]
        return torch.mean(output, dim=2)

    def forward(self, x):
        B, T, N = x.shape

        # 1. Conv1D时序特征提取
        x_conv = x.permute(0, 2, 1).unsqueeze(-1)  # [B, N, T, 1]
        x_conv = x_conv.reshape(B * N, 1, T)  # [B*N, 1, T]

        # 残差连接
        residual = self.residual_proj(x_conv)  # [B*N, conv_hidden, T]
        conv_out = self.conv1d(x_conv)  # [B*N, conv_hidden, T]
        fused_out = conv_out + residual  # [B*N, conv_hidden, T]

        # 使用全局时序特征构建图
        node_rep = torch.mean(fused_out, dim=-1)  # [B*N, conv_hidden] (时间维度平均)
        node_rep = node_rep.view(B, N, -1)  # [B, N, conv_hidden]

        # 2. 动态图构建
        laplacian = self.build_dynamic_graph(node_rep)
        supports = self.chebyshev_polynomials(laplacian)

        # 3. 图卷积处理
        # 输入维度: [B, N, T]
        conv_features = self.graph_convolution(x.permute(0, 2, 1), supports)  # [B, spatial_dim, T]

        # 时空特征全局池化
        spatial_global = self.global_pool(conv_features).squeeze(-1)  # [B, spatial_dim]

        # 4. Conv1D特征全局聚合
        conv_global = torch.mean(
            fused_out.view(B, N, self.conv_hidden, T),
            dim=(1, 3)  # 同时聚合节点和时间维度
        )  # [B, conv_hidden]

        # 5. 特征融合
        fused_features = torch.cat([spatial_global, conv_global], dim=-1)  # [B, spatial_dim + conv_hidden]
        output = self.fusion_layer(fused_features)  # [B, spatial_dim]


        return output  # 输出形状 [B, D] (D=spatial_dim)
        # return spatial_global


class SpatioTemporalModel00(nn.Module):
    def __init__(self, N=14, T=30, D=64):
        super().__init__()
        self.N = N
        self.T = T
        self.D = D

        # 空间分支
        # 使用Conv1D版特征提取器
        self.spatial_extractor = Conv1DDynamicGraphSpatialExtractor02(num_nodes=N, spatial_dim=D)

        # 时间分支
        self.time_proj = nn.Linear(N, D)
        self.pos_enc = PositionalEncoding00(D)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, 1, dim_feedforward=256), 2)
        # 时空交互
        self.space_cross = nn.MultiheadAttention(D, 8)
        self.time_cross = nn.MultiheadAttention(D, 8)
        self.fusion_gate = nn.Sequential(nn.Linear(2 * D, D), nn.Sigmoid())
        # 回归头
        self.mlp = nn.Sequential(
            nn.Linear(D, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # 初始化权重
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.space_cross.in_proj_weight)
        nn.init.xavier_uniform_(self.time_cross.in_proj_weight)
        for layer in self.transformer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
        # 注意力池化层 (替换平均池化)
        self.attention_pool = AttentionPooling(D)

    def forward_time(self, x):
        """时序特征提取流程"""
        # 特征投影 [B, T, N] -> [B, T, D]
        x = self.time_proj(x)
        # 位置编码增强时序感知
        x = self.pos_enc(x)
        # 调整维度适配Transformer: [B, T, D] -> [T, B, D]
        x = x.permute(1, 0, 2)
        # Transformer编码
        x = self.transformer(x)
        # 恢复维度: [T, B, D] -> [B, T, D]
        x = x.permute(1, 0, 2)
        # 注意力池化 (返回加权特征和权重)
        pooled, attn_weights = self.attention_pool(x)
        return pooled, attn_weights

    def cross_interact(self, space_feat, time_feat):
        # 双向交叉注意力
        s = space_feat.unsqueeze(0)  # [1,B,D]
        t = time_feat.unsqueeze(0)  # [1,B,D]
        # 空间-时间交叉
        s_ctx, _ = self.space_cross(s, t, t)  # [1,B,D]
        # 时间-空间交叉
        t_ctx, _ = self.time_cross(t, s, s)  # [1,B,D]
        # 门控融合
        fused = torch.cat([s_ctx, t_ctx], -1).squeeze(0)
        gate = self.fusion_gate(fused)
        return gate * (s_ctx + t_ctx).squeeze(0)

    def forward(self, x):
        # 空间特征提取
        spatial = self.spatial_extractor(x)  # [B,D]

        # 时间特征提取
        temporal, attn_weights = self.forward_time(x)  # [B,D]
        # 时空交互
        fused = self.cross_interact(spatial, temporal)
        # 回归预测
        return self.mlp(fused)  # [B,1]




if __name__ == "__main__":
    data_path_0 = "D:/software/Python/Pycharm/Py_workplace/python_learn/可运行的RUL开源代码/自己设计的/raw_data/"
    # data_path_0 = "/root/时空图/自己设计的/raw_data/"
    subset_name = "003"
    data_path = "D:/software/Python/Pycharm/Py_workplace/python_learn/可运行的RUL开源代码/自己设计的/raw_data/processed/"
    # data_path = "/root/时空图/自己设计的/raw_data/processed/"
    sensors = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
    window_size = 75
    slide_step = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # process_and_save_data(data_path_0, subset_name)   # txt生成csv
    # train_data, test_data = cluster(data_path, subset_name)
    # oc_history_cols(data_path, subset_name, train_data, test_data, save=True)

    train, test, val, scalingparams, rul_data, train02 = prepare_data(data_path, subset_name, sensors, window_size, slide_step)
    train_loader = DataLoader(train, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    unshuffle_train_loader = DataLoader(train, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    final_test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    val_loader = DataLoader(val, batch_size=512, shuffle=True, num_workers=0)
    train02 = DataLoader(train02, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    model = SpatioTemporalModel00().to(device)
    print(subset_name)
    # # **训练循环**
    # epoch = 1
    # temp_values = []
    # for i in range(epoch):
    #     for step, (x, y) in enumerate(unshuffle_train_loader):
    #         x, y = x.to(device, non_blocking=True, dtype=torch.float32), y.to(device, non_blocking=True, dtype=torch.float32)
    #         print(f"y shape: v{y}")
    #         out = model(x)
    #         print(f"out shape: {out}")
    #         # print(f"forecast shape: {forecast.shape}")# [1024, 1, 14][batch_size, horizon, units]
    #         # print(f"attention shape: {attention.shape}")# [14, 14][units, units]
    #         # print(f"foreback shape: {recon.shape}")# [1024, 14, 30][batch_size, time_step, units]
    #         break



    # **实例化 Train_Module**
    trainer = Train_Module(model, train_loader, val_loader, unshuffle_train_loader, final_test_loader, train02, subset_name, device="cuda")
    # # **训练**
    train_losses, eval_losses = trainer.train_model(epoch=80, lr=1e-3, loss_func=RMSELoss(), early_stop=3)

    # # **恢复最佳模型权重**
    # model.load_state_dict(torch.load('D:/software/Python/Pycharm/Py_workplace/python_learn/可运行的RUL开源代码/时空图论文/模型训练参数/best_model_FD003.pt'))  # 恢复最佳模型权重
    # model_weights_path = "D:/software/Python/Pycharm/Py_workplace/python_learn/可运行的RUL开源代码/时空图论文/模型训练参数/"
    #                      "best_model_FD003.pt"
    # model.load_state_dict(torch.load(model_weights_path))  # 恢复最佳模型权重
    # # **计算
    train_output, finaltest_output, engine_train_preds, engine_test_preds = trainer.compute_train_output()

    # # 读取 final_test_output.csv 文件
    # file_path = r"D:\software\Python\Pycharm\Py_workplace\python_learn\可运行的RUL开源代码\时空图论文\结果\engine_test_predictions.csv"
    # engine_test_preds = pd.read_csv(file_path)

    # train_predicted(data_path,  subset_name,  window_size,  train_output,
    #                 alpha_grid=0.5,  # 网格透明度，范围 0-1
    #                 alpha_high=0.8,  # 线条透明度，范围 0-1
    #                 _COLORS=["orange", "blue"]  # 颜色列表，分别对应预测 RUL 和实际 RUL 线条颜色
    #                 )

    # # 绘制测试集引擎的预测对比和误差图'
    # mae_test, rmse_test, max_error_test = plot_engine_comparison_combined_2(rul_data, engine_test_preds, subset_name, "Test Set: True vs Predicted RUL")
    #
    # file_paths = {
    #     'FD001': r"D:\software\Python\Pycharm\Py_workplace\python_learn\可运行的RUL开源代码\时空图论文\结果\001_errors.csv",
    #     'FD002': r"D:\software\Python\Pycharm\Py_workplace\python_learn\可运行的RUL开源代码\时空图论文\结果\002_errors.csv",
    #     'FD003': r"D:\software\Python\Pycharm\Py_workplace\python_learn\可运行的RUL开源代码\时空图论文\结果\003_errors.csv",
    #     'FD004': r"D:\software\Python\Pycharm\Py_workplace\python_learn\可运行的RUL开源代码\时空图论文\结果\004_errors.csv"
    # }
    # # 四色点图
    # stats = plot_error_distribution_from_csv(file_paths)
    # fianl_test_predicted(data_path, subset_name, window_size, finaltest_output, alpha_grid=0.5, alpha_high=0.8, _COLORS=["orange", "blue"])
    # # 计算最终测试评分
    score = calculate_standard_score(data_path, subset_name, window_size, finaltest_output)



