import torch
import torch.nn as nn
import cv2
import numpy as np

class IDMModule(nn.Module):

    def __init__(self):
        super(IDMModule, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU()
        )

        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.avg_pool3 = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(16, 3)
        self.fc2 = nn.Linear(16 * 4, 3)
        self.fc3 = nn.Linear(16 * 16, 3)

    def forward(self, x):

        fa = self.conv_block(x)

        fb = self.depthwise_conv(fa)

        f1 = self.avg_pool1(fb).view(fb.size(0), -1)  # Shape: (Batch, 16)
        f2 = self.avg_pool2(fb).view(fb.size(0), -1)  # Shape: (Batch, 16*4)
        f3 = self.avg_pool3(fb).view(fb.size(0), -1)  # Shape: (Batch, 16*16)

        phi_r1, phi_g1, phi_b1 = torch.split(self.fc1(f1), 1, dim=1)  # (Batch, 1)
        phi_r2, phi_g2, phi_b2 = torch.split(self.fc2(f2), 1, dim=1)  # (Batch, 1)
        phi_r3, phi_g3, phi_b3 = torch.split(self.fc3(f3), 1, dim=1)  # (Batch, 1)

        i_r, i_g, i_b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        i_gray1 = phi_r1.view(-1, 1, 1, 1) * i_r + phi_g1.view(-1, 1, 1, 1) * i_g + phi_b1.view(-1, 1, 1, 1) * i_b
        i_gray2 = phi_r2.view(-1, 1, 1, 1) * i_r + phi_g2.view(-1, 1, 1, 1) * i_g + phi_b2.view(-1, 1, 1, 1) * i_b
        i_gray3 = phi_r3.view(-1, 1, 1, 1) * i_r + phi_g3.view(-1, 1, 1, 1) * i_g + phi_b3.view(-1, 1, 1, 1) * i_b


        i_gray = (i_gray1 + i_gray2 + i_gray3) / 3  # Average the grayscale channels

        i_gray_np = i_gray.squeeze().cpu().detach().numpy()  # Shape (8, 256, 256)

        if len(i_gray_np.shape) == 2:
            batch_size = 1
            i_gray_np = np.expand_dims(i_gray_np, axis=0)
        elif len(i_gray_np.shape) == 3:
            batch_size = i_gray_np.shape[0]
        else:
            raise ValueError("Unexpected input shape of IDM module")

        base_layers = []
        detail_layers = []
        combined_images = []

        for i in range(batch_size):
            img = i_gray_np[i, :, :]

            kernel_size = (21, 21)
            base_layer = cv2.GaussianBlur(img, kernel_size, 0)
            detail_layer = cv2.subtract(img, base_layer)

            base_layer_tensor = torch.tensor(base_layer, dtype=i_gray.dtype).unsqueeze(0).unsqueeze(0).to(i_gray.device)
            detail_layer_tensor = torch.tensor(detail_layer, dtype=i_gray.dtype).unsqueeze(0).unsqueeze(0).to(i_gray.device)

            base_layers.append(base_layer_tensor)
            detail_layers.append(detail_layer_tensor)

            combined_image = torch.cat([base_layer_tensor, i_gray[i:i + 1], detail_layer_tensor],
                                       dim=1)
            combined_images.append(combined_image)

        i_combined = torch.cat(combined_images, dim=0)
        return i_combined

