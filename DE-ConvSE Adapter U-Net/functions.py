import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ================ 数据集划分 ================
def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

# ================ 显示预测结果 ================
def visualize(sample_y, out_y, error, s, y_mean, y_std):
    y_mean = y_mean.cpu().detach().numpy()
    y_std = y_std.cpu().detach().numpy()

    # 对 sample_y 和 out_y 进行逆标准化
    sample_y_denorm = sample_y * y_std + y_mean
    out_y_denorm = out_y * y_std + y_mean
    error_denorm = error * y_std

    # 获取样本的最小值和最大值
    minu = np.min(sample_y_denorm[s, 0, :, :])
    maxu = np.max(sample_y_denorm[s, 0, :, :])

    mint = np.min(sample_y_denorm[s, 1, :, :])
    maxt = np.max(sample_y_denorm[s, 1, :, :])

    minp = np.min(sample_y_denorm[s, 2, :, :])
    maxp = np.max(sample_y_denorm[s, 2, :, :])

    # 误差的最小值和最大值
    mineu = np.min(error_denorm[s, 0, :, :])
    maxeu = np.max(error_denorm[s, 0, :, :])

    minet = np.min(error_denorm[s, 1, :, :])
    maxet = np.max(error_denorm[s, 1, :, :])

    minep = np.min(error_denorm[s, 2, :, :])
    maxep = np.max(error_denorm[s, 2, :, :])

    # 绘图
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)

    # 绘制速度图
    plt.subplot(3, 3, 1)
    plt.title('CFD', fontsize=18)
    plt.imshow(np.transpose(sample_y_denorm[s, 0, :, :]), cmap='jet', vmin=minu, vmax=maxu, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Velocity', fontsize=18)

    # 绘制 U-Net 输出速度图
    plt.subplot(3, 3, 2)
    plt.title('U-Net', fontsize=18)
    plt.imshow(np.transpose(out_y_denorm[s, 0, :, :]), cmap='jet', vmin=minu, vmax=maxu, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')

    # 绘制误差图
    plt.subplot(3, 3, 3)
    plt.title('Error', fontsize=18)
    plt.imshow(np.transpose(error_denorm[s, 0, :, :]), cmap='jet', vmin=mineu, vmax=maxeu, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')

    # 绘制温度图
    plt.subplot(3, 3, 4)
    plt.imshow(np.transpose(sample_y_denorm[s, 1, :, :]), cmap='jet', vmin=mint, vmax=maxt, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Temperature', fontsize=18)

    # 绘制 U-Net 输出温度图
    plt.subplot(3, 3, 5)
    plt.imshow(np.transpose(out_y_denorm[s, 1, :, :]), cmap='jet', vmin=mint, vmax=maxt, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')

    # 绘制温度误差图
    plt.subplot(3, 3, 6)
    plt.imshow(np.transpose(error_denorm[s, 1, :, :]), cmap='jet', vmin=minet, vmax=maxet, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')

    # 绘制压力图
    plt.subplot(3, 3, 7)
    plt.imshow(np.transpose(sample_y_denorm[s, 2, :, :]), cmap='jet', vmin=minp, vmax=maxp, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Pressure', fontsize=18)

    # 绘制 U-Net 输出压力图
    plt.subplot(3, 3, 8)
    plt.imshow(np.transpose(out_y_denorm[s, 2, :, :]), cmap='jet', vmin=minp, vmax=maxp, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')

    # 绘制压力误差图
    plt.subplot(3, 3, 9)
    plt.imshow(np.transpose(error_denorm[s, 2, :, :]), cmap='jet', vmin=minep, vmax=maxep, origin='lower',
               extent=[0, 370, 0, 127], interpolation='bilinear')
    plt.colorbar(orientation='horizontal')

    plt.tight_layout()

    # 保存为高清图片
    plt.savefig(f"flow_field_{s}.png", bbox_inches="tight", dpi=300)
    plt.show()