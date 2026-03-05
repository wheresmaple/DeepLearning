import torch
import torch.nn as nn
import torchvision.models as models


# RaGAN损失定义
class RaGANLoss(nn.Module):
    def __init__(self):
        super(RaGANLoss, self).__init__()

    def forward(self, real_pred, fake_pred, is_discriminator=True):
        # 判别器损失: L_D = -E[log(D(R,F))] - E[log(1-D(F,R))]
        if is_discriminator:
            loss = -torch.mean(torch.log(real_pred + 1e-8)) - torch.mean(torch.log(1 - fake_pred + 1e-8))
        # 生成器损失: L_G = -E[log(1-D(R,F))] - E[log(D(F,R))]
        else:
            loss = -torch.mean(torch.log(1 - real_pred + 1e-8)) - torch.mean(torch.log(fake_pred + 1e-8))
        return loss


# VGG19感知损失（加载本地权重）
class VGGLoss(nn.Module):
    def __init__(self, feature_layer=35, device='cuda'):
        super(VGGLoss, self).__init__()
        # 加载本地VGG19权重，避免下载
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth', map_location=device))  # 根目录权重文件
        vgg_features = vgg.features[:feature_layer]
        # 冻结参数
        for param in vgg_features.parameters():
            param.requires_grad = False
        self.vgg_features = vgg_features.to(device)
        self.mse = nn.MSELoss()

    def forward(self, fake_img, real_img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(fake_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(fake_img.device)

        fake_img = (fake_img - mean) / std
        real_img = (real_img - mean) / std

        fake_feat = self.vgg_features(fake_img)
        real_feat = self.vgg_features(real_img)
        return self.mse(fake_feat, real_feat)