import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DualBranchFPN(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels):

        super(DualBranchFPN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1)

        self.attn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        feat1 = self.conv1(feat1)
        feat2 = self.conv2(feat2)
        fused = torch.cat([feat1, feat2], dim=1)
        attn_weights = self.attn(fused)
        fused_feat = feat1 * attn_weights + feat2 * (1 - attn_weights)

        return fused_feat


class FeaturePyramid(nn.Module):

    def __init__(self, out_channels, use_smoothing=True):
        super(FeaturePyramid, self).__init__()
        self.use_smoothing = use_smoothing

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1) for _ in range(4)
        ])

        if self.use_smoothing:
            self.smooth_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(4)
            ])
        else:
            self.smooth_convs = None

    def forward(self, fused_feats):

        P4, P3, P2, P1 = fused_feats
        up_P4 = F.interpolate(P4, size=P3.shape[2:], mode='nearest')
        P3 = P3 + self.lateral_convs[1](up_P4)
        up_P3 = F.interpolate(P3, size=P2.shape[2:], mode='nearest')
        P2 = P2 + self.lateral_convs[2](up_P3)
        up_P2 = F.interpolate(P2, size=P1.shape[2:], mode='nearest')
        P1 = P1 + self.lateral_convs[3](up_P2)

        if self.use_smoothing:
            P4 = self.smooth_convs[0](P4)
            P3 = self.smooth_convs[1](P3)
            P2 = self.smooth_convs[2](P2)
            P1 = self.smooth_convs[3](P1)

        return [P4, P3, P2, P1]



class FusionModule(nn.Module):


    def __init__(self, fpn_channels=[(128, 128), (256, 256), (512, 512), (1024, 1024)], out_channels=512, use_attention=True, use_smoothing=True):

        super(FusionModule, self).__init__()
        self.use_attention = use_attention

        self.fpn_fusion = nn.ModuleList()
        for (c1, c2) in fpn_channels:
            self.fpn_fusion.append(DualBranchFPN(c1, c2, out_channels))

        self.feature_pyramid = FeaturePyramid(out_channels, use_smoothing=use_smoothing)

        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1)

    def forward(self, feats1_list, feats2_list):

        feats1 = [feats1_list[0].view(-1, feats1_list[0].size(2), int(math.sqrt(feats1_list[0].size(1))), int(math.sqrt(feats1_list[0].size(1))) ),
                  feats1_list[1].view(-1, feats1_list[1].size(2), int(math.sqrt(feats1_list[1].size(1))), int(math.sqrt(feats1_list[1].size(1))) ),
                  feats1_list[2].view(-1, feats1_list[2].size(2), int(math.sqrt(feats1_list[2].size(1))), int(math.sqrt(feats1_list[2].size(1))) ),
                  feats1_list[3].view(-1, feats1_list[3].size(2), int(math.sqrt(feats1_list[3].size(1))), int(math.sqrt(feats1_list[3].size(1))) )]

        feats2 = [feats2_list[0].view(-1, feats2_list[0].size(2), int(math.sqrt(feats2_list[0].size(1))), int(math.sqrt(feats2_list[0].size(1))) ),
                  feats2_list[1].view(-1, feats2_list[1].size(2), int(math.sqrt(feats2_list[1].size(1))), int(math.sqrt(feats2_list[1].size(1))) ),
                  feats2_list[2].view(-1, feats2_list[2].size(2), int(math.sqrt(feats2_list[2].size(1))), int(math.sqrt(feats2_list[2].size(1))) ),
                  feats2_list[3].view(-1, feats2_list[3].size(2), int(math.sqrt(feats2_list[3].size(1))), int(math.sqrt(feats2_list[3].size(1))) )]

        fused_feats = []
        for i in range(len(self.fpn_fusion)):
            fused = self.fpn_fusion[i](feats1[i], feats2[i])
            fused_feats.append(fused)

        fused_feats_reversed = fused_feats[::-1]

        pyramid_feats = self.feature_pyramid(fused_feats_reversed)
        P3_down = F.interpolate(pyramid_feats[1], size=pyramid_feats[0].shape[2:], mode='nearest')
        P2_down = F.interpolate(pyramid_feats[2], size=pyramid_feats[0].shape[2:], mode='nearest')
        P1_down = F.interpolate(pyramid_feats[3], size=pyramid_feats[0].shape[2:], mode='nearest')

        combined = torch.cat([pyramid_feats[0], P3_down, P2_down, P1_down], dim=1)
        combined = self.final_conv(combined)

        B, C, H, W = combined.size()
        final_feat = combined.view(B, C, H * W).permute(0, 2, 1)

        return final_feat

