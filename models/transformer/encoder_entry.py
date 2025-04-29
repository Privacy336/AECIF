import torch
import numpy as np
from models.transformer.swintransformer import SwinTransformer
from models.transformer.Multiswintransformer import MultiSwinTransformer

# from models.transformer.visiontransformer import VisionTransformer

# import ml_collections

def build_encoder(config):
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=21841,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.5,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )

    swin_resume_path = config.swin_resume_path
    checkpoint = torch.load(swin_resume_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    return model


def build_encoder_MulitSwinTransformer(config):
    model = MultiSwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=21841,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.5,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )

    swin_resume_path = config.swin_resume_path
    checkpoint = torch.load(swin_resume_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    return model


def build_random_init_encoder(config):
    model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=21841,
                            embed_dim=128,
                            depths=[ 2, 2, 18, 2 ],
                            num_heads=[ 4, 8, 16, 32 ],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.5,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False)
    swin_resume_path = config.swin_resume_path
    checkpoint = torch.load(swin_resume_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)


    return model

