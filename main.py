import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

# 导入自定义模块
from models._init_ import ESRGAN, RaGANLoss, VGGLoss
from data._init_ import build_dataloaders


# 训练函数
def train_model(model, train_loader, criterion_gan, criterion_vgg, criterion_l1,
                optimizer_g, optimizer_d, epochs=50, device='cuda'):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        total_d_loss, total_g_loss = 0.0, 0.0
        for lr_imgs, hr_imgs in pbar:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # -------------------------- 训练判别器 --------------------------
            optimizer_d.zero_grad()
            fake_imgs = model(lr_imgs)
            # RaGAN判别器输入: D(HR, Fake) 和 D(Fake, HR)
            real_pred = model.discriminate(hr_imgs, fake_imgs)
            fake_pred = model.discriminate(fake_imgs.detach(), hr_imgs)
            loss_d = criterion_gan(real_pred, fake_pred, is_discriminator=True)
            loss_d.backward()
            optimizer_d.step()
            total_d_loss += loss_d.item()

            # -------------------------- 训练生成器 --------------------------
            optimizer_g.zero_grad()
            fake_imgs = model(lr_imgs)
            real_pred = model.discriminate(hr_imgs, fake_imgs)
            fake_pred = model.discriminate(fake_imgs, hr_imgs)
            loss_gan = criterion_gan(real_pred, fake_pred, is_discriminator=False)
            loss_vgg = criterion_vgg(fake_imgs, hr_imgs)  # VGG感知损失
            loss_l1 = criterion_l1(fake_imgs, hr_imgs)
            # 生成器总损失: 对抗损失 + VGG损失 + L1损失
            loss_g = loss_gan + 0.006 * loss_vgg + 0.01 * loss_l1
            loss_g.backward()
            optimizer_g.step()
            total_g_loss += loss_g.item()

            pbar.set_postfix({
                'D Loss': loss_d.item(),
                'G Loss': loss_g.item(),
                'Avg D Loss': total_d_loss / (pbar.n + 1),
                'Avg G Loss': total_g_loss / (pbar.n + 1)
            })


# 测试函数
def test_model(model, test_loaders, device='cuda'):
    model.eval()
    psnr_results = {}
    with torch.no_grad():
        for name, loader in test_loaders.items():
            psnr_list = []
            for lr_imgs, hr_imgs in tqdm(loader, desc=f'Testing {name}'):
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                fake_imgs = model(lr_imgs)

                # 转换为numpy计算PSNR
                fake_np = fake_imgs.squeeze(0).permute(1, 2, 0).cpu().numpy()
                hr_np = hr_imgs.squeeze(0).permute(1, 2, 0).cpu().numpy()
                psnr_val = psnr(hr_np, fake_np, data_range=1.0)
                psnr_list.append(psnr_val)
            psnr_results[name] = np.mean(psnr_list)
    return psnr_results


if __name__ == '__main__':
    # 1. 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 2. 构建数据加载器
    print('加载数据集...')
    train_loader, test_loaders = build_dataloaders(batch_size=8)

    # 3. 初始化模型（原版+改进版）
    print('初始化模型...')
    esrgan_original = ESRGAN(use_improved=False).to(device)
    esrgan_improved = ESRGAN(use_improved=True).to(device)

    # 4. 损失函数与优化器
    criterion_gan = RaGANLoss()
    criterion_vgg = VGGLoss(device=device)  # 加载本地VGG19权重
    criterion_l1 = nn.L1Loss()

    # 原版模型优化器
    optimizer_g_ori = torch.optim.Adam(esrgan_original.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_d_ori = torch.optim.Adam(esrgan_original.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    # 改进版模型优化器
    optimizer_g_imp = torch.optim.Adam(esrgan_improved.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_d_imp = torch.optim.Adam(esrgan_improved.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # 5. 训练模型
    print('\n===== 训练原版ESRGAN =====')
    train_model(esrgan_original, train_loader, criterion_gan, criterion_vgg, criterion_l1,
                optimizer_g_ori, optimizer_d_ori, epochs=5, device=device)

    print('\n===== 训练改进版ESRGAN =====')
    train_model(esrgan_improved, train_loader, criterion_gan, criterion_vgg, criterion_l1,
                optimizer_g_imp, optimizer_d_imp, epochs=5, device=device)

    # 6. 测试模型并计算PSNR
    print('\n===== 测试原版ESRGAN =====')
    psnr_original = test_model(esrgan_original, test_loaders, device=device)
    print('\n===== 测试改进版ESRGAN =====')
    psnr_improved = test_model(esrgan_improved, test_loaders, device=device)

    # 7. 输出对比结果
    print('\n================ PSNR对比结果 ================')
    print(f'数据集\t\t原版ESRGAN (dB)\t改进版ESRGAN (dB)\t提升值 (dB)')
    print('-' * 60)
    for dataset in psnr_original.keys():
        ori = psnr_original[dataset]
        imp = psnr_improved[dataset]
        delta = imp - ori
        print(f'{dataset}\t\t{ori:.4f}\t\t{imp:.4f}\t\t{delta:.4f}')

    # 8. 保存模型（可选）
    torch.save(esrgan_original.state_dict(), './esrgan_original.pth')
    torch.save(esrgan_improved.state_dict(), './esrgan_improved.pth')
    print('\n模型已保存至当前目录！')