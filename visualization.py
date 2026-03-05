import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from data._init_ import ImageDataset, UnderwaterDataset, build_dataloaders

# 导入实际的模型结构（关键修复点）
from models._init_ import ESRGAN as RealESRGAN
from models.blocks import RR_MADB, DenseBlock, MADB, ECA

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def load_esrgan_model(weight_path, use_improved=True, scale=2):
    """加载实际训练的ESRGAN模型权重（修复结构不匹配问题）"""
    # 初始化实际的模型结构
    model = RealESRGAN(
        use_improved=use_improved,
        in_channels=3,
        num_blocks=16,
        growth_rate=64
    ).to(device)

    # 加载权重（处理多GPU训练的情况）
    state_dict = torch.load(weight_path, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # 移除module.前缀（如果存在）
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 关键修复：过滤掉判别器的参数（仅保留生成器参数）
    generator_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('discriminator.')}

    # 仅加载生成器参数（strict=False兼容其他微小差异）
    model.load_state_dict(generator_state_dict, strict=False)

    # 冻结所有参数（推理阶段无需梯度）
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
# 修复：图像反归一化逻辑（适配实际模型输出范围）
def visualize_sr_results(lr_img, hr_img, sr_img_improved, sr_img_original, save_path=None):
    """
    可视化超分结果对比（修复反归一化逻辑）
    lr_img: 低分辨率图像
    hr_img: 高分辨率参考图像
    sr_img_improved: 改进版ESRGAN重建图像
    sr_img_original: 原版ESRGAN重建图像
    save_path: 保存路径（可选）
    """

    # 修复：tensor转图像（适配实际模型输出，增加数值范围校验）
    def tensor2img(tensor):
        img = tensor.cpu().detach().permute(1, 2, 0).numpy()
        # 修复：先归一化到0-1（解决数值溢出/下溢问题）
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    # 转换图像格式
    lr_img = tensor2img(lr_img)
    hr_img = tensor2img(hr_img)
    sr_img_improved = tensor2img(sr_img_improved)
    sr_img_original = tensor2img(sr_img_original)

    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))


    # 第一行：LR、HR、改进版ESRGAN
    axes[0, 0].imshow(lr_img)
    axes[0, 0].set_title(f'低分辨率图像\n尺寸: {lr_img.shape[1]}x{lr_img.shape[0]}', fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(hr_img)
    axes[0, 1].set_title(f'高分辨率参考图像\n尺寸: {hr_img.shape[1]}x{hr_img.shape[0]}', fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(sr_img_improved)
    axes[0, 2].set_title(f'改进版ESRGAN重建\n尺寸: {sr_img_improved.shape[1]}x{sr_img_improved.shape[0]}', fontsize=14)
    axes[0, 2].axis('off')



    axes[1, 2].imshow(sr_img_original)
    axes[1, 2].set_title(f'原版ESRGAN重建\n尺寸: {sr_img_original.shape[1]}x{sr_img_original.shape[0]}', fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果已保存至: {save_path}")
    plt.show()


# 修复：定性分析函数（适配实际模型前向传播）
def qualitative_analysis(test_loader, model_improved, model_original, dataset_name, num_samples=5):
    """
    对测试集进行定性分析（修复模型前向传播逻辑）
    test_loader: 测试集数据加载器
    model_improved: 改进版ESRGAN模型
    model_original: 原版ESRGAN模型
    dataset_name: 数据集名称
    num_samples: 可视化的样本数量
    """
    # 创建保存目录
    save_dir = f'./sr_results/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (lr_img, hr_img) in enumerate(test_loader):
            if idx >= num_samples:
                break

            # 图像移至设备
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # 修复：前向传播仅取生成器输出（避免判别器干扰）
            sr_img_improved = model_improved(lr_img)[0]  # 改进版模型输出
            sr_img_original = model_original(lr_img)[0]  # 原版模型输出

            # 修复：调整HR图像尺寸（使用align_corners=True匹配模型插值方式）
            hr_img = F.interpolate(
                hr_img,
                size=sr_img_improved.shape[-2:],
                mode='bicubic',
                align_corners=True  # 关键：与模型训练时的插值方式一致
            )[0]

            # 可视化并保存结果
            save_path = os.path.join(save_dir, f'sample_{idx + 1}.png')
            visualize_sr_results(
                lr_img[0],
                hr_img,
                sr_img_improved,
                sr_img_original,
                save_path=save_path
            )

            print(f"已处理 {dataset_name} 数据集第 {idx + 1} 个样本")


# 主执行流程（修复模型加载参数）
if __name__ == '__main__':
    # 1. 加载预训练模型（指定use_improved参数）
    weight_paths = {
        'improved': r'D:\Code\DeepLearning\esrgan_improved.pth',
        'original': r'D:\Code\DeepLearning\esrgan_original.pth'
    }

    # 修复：加载改进版/原版模型时指定正确的参数
    model_improved = load_esrgan_model(weight_paths['improved'], use_improved=True)
    model_original = load_esrgan_model(weight_paths['original'], use_improved=False)

    print("模型加载完成！")

    # 2. 构建测试集加载器
    _, test_loaders = build_dataloaders(batch_size=1, target_size=(256, 256))

    # 3. 选择要分析的测试集
    datasets_to_analyze = ['Set5', 'Set14', 'BSD100', 'Underwater']

    # 4. 对每个测试集进行定性分析
    for dataset_name in datasets_to_analyze:
        print(f"\n开始分析 {dataset_name} 数据集...")
        test_loader = test_loaders[dataset_name]
        qualitative_analysis(
            test_loader=test_loader,
            model_improved=model_improved,
            model_original=model_original,
            dataset_name=dataset_name,
            num_samples=3  # 每个数据集可视化3个样本
        )

    print("\n定性分析完成！所有结果已保存至 ./sr_results 目录")