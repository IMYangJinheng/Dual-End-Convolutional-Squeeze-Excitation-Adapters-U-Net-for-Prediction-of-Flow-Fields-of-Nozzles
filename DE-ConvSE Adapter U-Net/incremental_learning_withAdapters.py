import os
import torch
import pickle
import json
from torch.utils.data import TensorDataset, DataLoader


def load_incremental_data(update_stats=True):
    """加载增量学习数据（小样本），读取旧参数并在训练后更新"""
    import os
    print("加载增量数据...")
    new_x = pickle.load(open("./Configuration/dataX.pkl", "rb"))
    new_y = pickle.load(open("./Configuration/dataY.pkl", "rb"))

    new_x = torch.FloatTensor(new_x)
    new_y = torch.FloatTensor(new_y)

    # ------------------------------
    # 1️⃣ 加载旧的标准化参数
    # ------------------------------
    if os.path.exists("./Configuration/x_mean.pt"):
        x_mean_old = torch.load("./Configuration/x_mean.pt")
        x_std_old = torch.load("./Configuration/x_std.pt")
        y_mean_old = torch.load("./Configuration/y_mean.pt")
        y_std_old = torch.load("./Configuration/y_std.pt")
        print("✅ 成功加载旧的标准化参数")
    else:
        # 如果第一次训练（没有旧参数）
        x_mean_old = new_x.mean(dim=0, keepdim=True)
        x_std_old = new_x.std(dim=0, keepdim=True) + 1e-8
        y_mean_old = new_y.mean(dim=0, keepdim=True)
        y_std_old = new_y.std(dim=0, keepdim=True)
        print("⚠️ 未检测到旧标准化参数，使用当前数据初始化")

    # ------------------------------
    # 2️⃣ 使用旧参数标准化新数据
    # ------------------------------
    new_x_norm = (new_x - x_mean_old) / x_std_old
    new_y_norm = (new_y - y_mean_old) / y_std_old

    print(f"增量数据形状: x={new_x_norm.shape}, y={new_y_norm.shape}")

    # ------------------------------
    # 3️⃣ 训练结束后可更新标准化参数
    # ------------------------------
    if update_stats:
        print("🔁 更新全局标准化参数（指数滑动平均）")

        # 计算新样本统计
        new_x_mean = new_x.mean(dim=0, keepdim=True)
        new_x_std = new_x.std(dim=0, keepdim=True) + 1e-8
        new_y_mean = new_y.mean(dim=0, keepdim=True)
        new_y_std = new_y.std(dim=0, keepdim=True) + 1e-8

        # 指数滑动平均（EMA）更新
        alpha = 0.2  # 可调：0.1~0.3 较稳健
        x_mean_updated = (1 - alpha) * x_mean_old + alpha * new_x_mean
        x_std_updated = (1 - alpha) * x_std_old + alpha * new_x_std
        y_mean_updated = (1 - alpha) * y_mean_old + alpha * new_y_mean
        y_std_updated = (1 - alpha) * y_std_old + alpha * new_y_std

        # 保存新参数
        torch.save(x_mean_updated, "./Configuration/x_mean.pt")
        torch.save(x_std_updated, "./Configuration/x_std.pt")
        torch.save(y_mean_updated, "./Configuration/y_mean.pt")
        torch.save(y_std_updated, "./Configuration/y_std.pt")

        print("💾 新标准化参数已保存 (融合旧参数 + 新数据统计)")

    return TensorDataset(new_x_norm, new_y_norm)


def create_adapter_model_correct():
    """创建正确的适配器模型 - 基于原始UNetEx结构"""
    from MY_Models.UNetEx import UNetEx

    # 加载原始模型配置
    with open("./Configuration/model_config.json", 'r') as f:
        config = json.load(f)

    print(f"🎯 创建基础模型: {len(config['filters'])}层结构")

    # 创建原始UNetEx模型
    base_model = UNetEx(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        filters=config['filters'],
        kernel_size=config['kernel_size'],
        layers=3,
        weight_norm=config['weight_norm'],
        batch_norm=config['batch_norm']
    )

    # 加载预训练权重
    base_model.load_state_dict(torch.load("./Configuration/model_complete.pth"))

    # 创建适配器包装器
    adapter_model = AdapterWrapper(base_model, adapter_reduction=4)

    return adapter_model, config


class AdapterWrapper(torch.nn.Module):
    """适配器包装器 - 在原始UNetEx基础上添加适配层"""

    def __init__(self, base_model, adapter_reduction=4):
        super().__init__()
        self.base_model = base_model

        # 冻结基础模型的所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 为每个编码器层添加适配器
        self.adapters = torch.nn.ModuleList()
        for i, encoder_block in enumerate(self.base_model.encoder):
            # 获取该层的输出通道数
            if hasattr(encoder_block[0], 'weight'):
                out_channels = encoder_block[0].weight.shape[0]
            else:
                # 估计通道数
                out_channels = [8, 16, 32, 32, 64, 64, 128][i]

            adapter = torch.nn.Sequential(
                torch.nn.Conv2d(out_channels, out_channels // adapter_reduction, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels // adapter_reduction, out_channels, 1)
            )
            self.adapters.append(adapter)

        print(f"🔧 添加了 {len(self.adapters)} 个适配器")

    def forward(self, x):
        # 使用基础模型的编码过程，但通过适配器
        tensors = []
        indices = []
        sizes = []

        # 编码过程（通过适配器）
        for i, encoder in enumerate(self.base_model.encoder):
            x = encoder(x)

            # 通过适配器
            if i < len(self.adapters):
                x = x + self.adapters[i](x)  # 残差连接

            sizes.append(x.size())
            tensors.append(x)
            x, ind = torch.nn.functional.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)

        # 使用基础模型的解码过程
        x = self.base_model.decode(x, tensors, indices, sizes)

        return x

    def get_adapter_parameters(self):
        """返回所有适配器参数（用于优化）"""
        return list(self.adapters.parameters())

    def train_adapters_only(self):
        """确保只有适配器参数可训练"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.adapters.parameters():
            param.requires_grad = True


def incremental_training_correct():
    """正确的小样本增量学习实现"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建正确的适配器模型
    model, config = create_adapter_model_correct()
    model.to(device)

    # 确保只有适配器可训练
    model.train_adapters_only()

    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 模型参数: 可训练 {trainable_params:,} / 总计 {total_params:,}")
    print(f"🔒 冻结比例: {(1 - trainable_params / total_params) * 100:.2f}%")

    # 加载增量数据
    incremental_dataset = load_incremental_data()

    # 仅优化适配器参数
    adapter_params = model.get_adapter_parameters()
    optimizer = torch.optim.AdamW(adapter_params, lr=5e-5, weight_decay=1e-5)  # 原文使用5e-5

    # 损失函数（与原始训练保持一致）
    channels_weights = torch.load("./Configuration/channels_weights.pt").to(device)

    def loss_func(output, y):
        losst = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        lossu = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        lossp = torch.abs((output[:, 2, :, :] - y[:, 2, :, :])).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        loss = (lossu + losst + lossp) / channels_weights
        return torch.sum(loss)

    # 训练前测试前向传播
    print("\n🧪 测试前向传播...")
    try:
        with torch.no_grad():
            test_batch = next(iter(DataLoader(incremental_dataset, batch_size=2)))
            test_x, test_y = test_batch
            test_x, test_y = test_x.to(device), test_y.to(device)
            test_output = model(test_x)
            print(f"✅ 前向传播测试成功!")
            print(f"   输入: {test_x.shape} -> 输出: {test_output.shape}")
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        return None, []

    # 训练循环（按照原文参数）
    print("\n🎯 开始小样本增量训练（Adapter Tuning）...")
    print("   配置: lr=5e-5, batch_size=2, epochs=500")

    model.train()
    train_losses = []
    best_loss = float('inf')
    patience = 50
    patience_counter = 0

    epochs = 1000
    batch_size = 2  # 原文使用batch_size=2

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0

        for batch in DataLoader(incremental_dataset, batch_size=batch_size, shuffle=True):
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / len(incremental_dataset)
        train_losses.append(avg_loss)

        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "./best_adapter_model.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0 or (epoch + 1) <= 10:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        if patience_counter >= patience:
            print(f"🛑 早停触发于第 {epoch + 1} 轮")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load("./best_adapter_model.pth"))
    print(f"\n✅ 小样本增量学习完成！最佳损失: {best_loss:.6f}")

    # 评估
    print("\n📊 在新构型上评估模型...")
    model.eval()
    with torch.no_grad():
        test_x, test_y = incremental_dataset[:]
        test_x, test_y = test_x.to(device), test_y.to(device)

        predictions = model(test_x)
        error = torch.abs(predictions - test_y)

        mse = torch.mean(error ** 2)
        mae = torch.mean(error)

        # 通道级别的评估
        mse_channels = torch.mean(error ** 2, dim=(0, 2, 3))
        mae_channels = torch.mean(error, dim=(0, 2, 3))

        print("评估结果:")
        print(f"  MSE: {mse.item():.6f}")
        print(f"  MAE: {mae.item():.6f}")
        print(f"  通道 MSE: [T: {mse_channels[0]:.6f}, U: {mse_channels[1]:.6f}, P: {mse_channels[2]:.6f}]")
        print(f"  通道 MAE: [T: {mae_channels[0]:.6f}, U: {mae_channels[1]:.6f}, P: {mae_channels[2]:.6f}]")

        # =========================
        # ✅ 可视化与模型保存
        # =========================
        print("\n📈 绘制训练损失曲线...")
        import matplotlib.pyplot as plt
        from functions import visualize  # 导入你的可视化函数

        # 绘制损失曲线
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Incremental Training Loss Curve', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./loss_curve.png", dpi=300)
        plt.show()
        print("✅ 训练损失曲线已保存为 loss_curve.png")

        # 保存最终模型
        torch.save(model.state_dict(), "./final_adapter_model.pth")
        print("💾 模型权重已保存到: ./final_adapter_model.pth")

        # =========================
        # ✅ CFD 可视化输出（3 张样例）
        # =========================
        try:
            print("\n🌈 绘制 3 张预测可视化样例...")

            # 加载标准化参数
            y_mean = torch.load("./Configuration/y_mean.pt").to(device)
            y_std = torch.load("./Configuration/y_std.pt").to(device)

            # 为防止 GPU 占用过多，先转回 CPU
            sample_y = test_y.cpu()
            out_y = predictions.cpu()
            error_map = error.cpu()

            # 绘制前三个样本
            for s in range(3):
                visualize(
                    sample_y=sample_y.unsqueeze(0),
                    out_y=out_y.unsqueeze(0),
                    error=error_map.unsqueeze(0),
                    s=s,
                    y_mean=y_mean,
                    y_std=y_std
                )
                print(f"✅ 可视化完成 flow_field_{s}.png")

        except Exception as e:
            print(f"⚠️ 可视化绘制失败: {e}")

    return model, train_losses


if __name__ == "__main__":
    print("=" * 60)
    print("          小样本增量学习 - 正确实现")
    print("=" * 60)
    print("📝 策略: 冻结基础模型，只训练适配层")
    print("📝 参数: lr=5e-5, batch_size=2, 早停机制")
    print("=" * 60)

    model, losses = incremental_training_correct()
    if model is not None:
        print("\n🎉 Adapter Tuning 成功完成！")
        if losses:
            print(f"   初始损失: {losses[0]:.6f}")
            print(f"   最终损失: {losses[-1]:.6f}")
    else:
        print("\n💥 增量学习失败！")