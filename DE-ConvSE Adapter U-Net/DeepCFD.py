import os
import json
import torch
import pickle
from torch.utils.data import TensorDataset
from MY_Models.UNetEx import UNetEx
from MY_Models.UNetEx_ResAtt import UNetEx_ResAtt
from matplotlib import pyplot as plt
from functions import *
from train_functions import *
import os

if __name__ == "__main__":
    #  ================ 加载数据集 ================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = pickle.load(open("./dataX.pkl", "rb"))
    y = pickle.load(open("./dataY.pkl", "rb"))

    #  ================ 检查数据类型并转换为 PyTorch 张量 ================
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    print("Shape of dataX:", x.shape)
    print("Shape of dataY:", y.shape)

    #  ================ 标准化数据 ================
    x_mean = x.mean()
    x_std = x.std()
    x = (x - x_mean) / x_std

    y_mean = y.mean(dim=(0, 2, 3), keepdim=True)
    y_std = y.std(dim=(0, 2, 3), keepdim=True)
    y = (y - y_mean) / y_std

    #  ================ 保存标准化参数（新增部分） ================
    print("正在保存标准化参数...")
    torch.save(x_mean, "./x_mean.pt")
    torch.save(x_std, "./x_std.pt")
    torch.save(y_mean, "./y_mean.pt")
    torch.save(y_std, "./y_std.pt")
    print("标准化参数已保存！")

    #  ================ 计算每个通道的权重 ================
    channels_weights = torch.sqrt(torch.mean(y.permute(0, 2, 3, 1).reshape((30 * 1110 * 381, 3)) ** 2, dim=0)).view(1, -1, 1, 1).to(device)
    print("Channels weights:", channels_weights)

    # 保存通道权重（新增部分）
    torch.save(channels_weights, "./channels_weights.pt")
    print("通道权重已保存！")

    #  ================ 设置结果的保存目录为.run ================
    simulation_directory = "./Run/"
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)

    #  ================ 切分数据集为训练集和测试集 ================
    train_data, test_data = split_tensors(x, y, ratio=0.8)
    train_dataset, test_dataset = TensorDataset(*train_data), TensorDataset(*test_data)
    test_x, test_y = test_dataset[:]
    torch.manual_seed(0)

    # ================ 超参数设置 ================
    lr = 0.0005
    kernel_size = 5
    filters = [8, 16, 32, 32, 64, 64, 128]
    bn = False
    wn = False
    wd = 0.005

    #  ================ 创建模型实例 ================
    model = UNetEx(in_channels=2, out_channels=3, filters=filters, kernel_size=kernel_size, batch_norm=bn,
                   weight_norm=wn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    #  ================ 记录损失用于后续分析 ================
    config = {}
    train_loss_curve = []
    test_loss_curve = []
    train_u_ape_curve = []
    test_u_ape_curve = []
    train_t_ape_curve = []
    test_t_ape_curve = []
    train_p_ape_curve = []
    test_p_ape_curve = []


    # ================ 定义损失函数 ================
    def loss_func(model, batch):
        x, y = batch
        output = model(x)
        losst = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        lossu = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        lossp = torch.abs((output[:, 2, :, :] - y[:, 2, :, :])).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        loss = (lossu + losst + lossp) / channels_weights
        return torch.sum(loss), output


    # ================ 训练 ================
    DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(model, loss_func, train_dataset,
                                                                              test_dataset, optimizer, epochs=1000,
                                                                              batch_size=4, device=device,
                                                                              m_uxape_name="U APE",
                                                                              m_uxape_on_batch=lambda scope: float(
                                                                                  torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 0, :, :] -
                                                                                      scope["output"][:, 0, :,
                                                                                      :])) / torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 0, :, :]))),
                                                                              m_uxape_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),

                                                                              m_uyape_name="T APE",
                                                                              m_uyape_on_batch=lambda scope: float(
                                                                                  torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 1, :, :] -
                                                                                      scope["output"][:, 1, :,
                                                                                      :])) / torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 1, :, :]))),
                                                                              m_uyape_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),

                                                                              m_pape_name="P APE",
                                                                              m_pape_on_batch=lambda scope: float(
                                                                                  torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 2, :, :] -
                                                                                      scope["output"][:, 2, :,
                                                                                      :])) / torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 2, :, :]))),
                                                                              m_pape_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(scope["dataset"])
                                                                              )

    # =============== 可视化预测结果 ================
    out = DeepCFD(test_x[:10].to(device))
    error = torch.abs(out.cpu() - test_y[:10].cpu())
    n_samples = min(10, test_y.shape[0])
    for s in range(n_samples):
        visualize(test_y[:n_samples].cpu().detach().numpy(),
                  out[:n_samples].cpu().detach().numpy(),
                  error[:n_samples].cpu().detach().numpy(),
                  s, y_std, y_mean)

    # ================ 保存模型 ================
    print("正在保存模型...请耐心等候")
    model_save_path = os.path.join("./model_complete.pth")
    print(f"正在保存模型到：{model_save_path}")
    try:
        torch.save(model.state_dict(), model_save_path)
        print("模型已保存！")

        # 保存训练配置信息（新增部分）
        config = {
            'filters': filters,
            'kernel_size': kernel_size,
            'batch_norm': bn,
            'weight_norm': wn,
            'in_channels': 2,
            'out_channels': 3
        }
        with open('./model_config.json', 'w') as f:
            json.dump(config, f)
        print("模型配置已保存！")

    except Exception as e:
        print(f"保存模型时发生错误: {e}")