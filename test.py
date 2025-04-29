import random
from data import TextField, RawField, ImageField
from data import Emphasis5k, DataLoader
from models.transformer import TransformerDecoderLayer, ImageEnhanceTransformer
from models.transformer.encoder_entry import build_encoder_MulitSwinTransformer
import torch
import argparse
import os
import pickle
import numpy as np
from models.grid_feature import Grid_feature_fuse_extractor
from models.transformer.dual_branch_swin_fpn_fusion import FusionModule
from models.transformer.IDMModule import IDMModule
from models.transformer import encoders
import json
from tqdm import tqdm
import evaluation
import sys
import itertools


def read_json(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)
        return json_data


def predict_captions_test(model, dataloader, text_field, save_dir, device):
    model.eval()
    gen = {}
    gts = {}
    dict = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images_list, caps_gt) in enumerate(iter(dataloader)):
            images = images_list[0].to(device)
            images_id_list = (images, images_list[1])
            with torch.no_grad():
                out, _ = model(mode='rl', img_id=images_id_list, max_len=20,
                               eos_idx=text_field.vocab.stoi['<eos>'],
                               beam_size=5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)

            images_file_id = images_id_list[1]
            for id, caps in zip(images_file_id, caps_gen):
                dict[id] = ' '.join(caps)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--dataset', type=str, default='Emphasis5k')
    parser.add_argument('--exp_name', type=str, default='train_grid_m2transformer')
    parser.add_argument('--swin_resume_path', type=str, default='', help='the file path of the pretrained Swin Transformer weights')
    parser.add_argument('--img_root_path', type=str, default='', help='the path to the underwater images')
    parser.add_argument('--annotation_folder', type=str, default='', help='annotation file')
    parser.add_argument('--save_folder', type=str, default='', help='file saving path')
    parser.add_argument('--Encoder_layer', type=int, default=3)
    parser.add_argument('--Decoder_layer', type=int, default=3)
    parser.add_argument('--gpu_number', type=int, nargs='+', default=[4, 6, 3, 7])
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--port', type=int, default=63000)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--val_test_batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--xe_least', type=int, default=15)  # 15
    parser.add_argument('--xe_most', type=int, default=20)  # 20
    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--refine_epoch_rl', type=int, default=28)
    parser.add_argument('--resume_last', type=bool, default=True)
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--IMG_SIZE', type=int, default=224)
    parser.add_argument('--COLOR_JITTER', type=float, default=0.4)
    parser.add_argument('--AUTO_AUGMENT', type=str, default='none', choices=['rand-m9-mstd0.5', 'none'])
    parser.add_argument('--REPROB', type=float, default=0.5)
    parser.add_argument('--REMODE', type=str, default='pixel')
    parser.add_argument('--RECOUNT', type=int, default=1)
    parser.add_argument('--INTERPOLATION', type=str, default='bicubic')
    parser.add_argument("--do_debug", action='store_true', help="Whether debug or not")
    args = parser.parse_args()
    print(args)

    print('SWIN-B Transformer Evaluation')

    device_a = torch.device('cuda:{}'.format(args.gpu_number[0]))
    device_b = torch.device('cuda:{}'.format(args.gpu_number[1]))
    device_c = torch.device('cuda:{}'.format(args.gpu_number[2]))
    device_d = torch.device('cuda:{}'.format(args.gpu_number[3]))

    device = torch.device(device_b)


    image_field = ImageField(config=args)
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)


    dataset = Emphasis5k(image_field, text_field, '/', args.annotation_folder, args.annotation_folder)

    train_dataser, val_dataset, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab_{}.pkl'.format(args.dataset), 'rb'))



    image_enhancement = IDMModule().to(device_a)
    backbone_1 = build_encoder_MulitSwinTransformer(config=args).to(device_b)
    backbone_2 = build_encoder_MulitSwinTransformer(config=args).to(device_c)
    fusion_model = FusionModule(out_channels=1024, use_attention=True, use_smoothing=True).to(device_d)
    grid_feature = Grid_feature_fuse_extractor(config=args).to(device_d)
    encoder = encoders.singal_grid_transformer(N=args.Encoder_layer, padding_idx=0, d_model=512, config=args).to(device_d)
    decoder = TransformerDecoderLayer(vocab_size=len(text_field.vocab), max_len=54, N_dec=args.Decoder_layer, padding_idx=text_field.vocab.stoi['<pad>']).to(device_d)
    model = ImageEnhanceTransformer(text_field.vocab.stoi['<bos>'],image_enhancement, backbone_1, backbone_2, fusion_model, grid_feature, encoder, decoder, config=args)

    data = torch.load(args.save_folder)
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.val_test_batch_size)
    scores = predict_captions_test(model = model,
                              dataloader = dict_dataloader_test,
                              text_field = text_field,
                              save_dir=args.save_folder,
                              device = device)
    print(scores)

