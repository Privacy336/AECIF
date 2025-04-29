import random
from data import TextField, RawField, ImageField
from data import Emphasis5k, DataLoader
from evaluation import PTBTokenizer, Cider
from models.transformer import TransformerDecoderLayer, ImageEnhanceTransformer
from models.transformer.encoder_entry import build_encoder, build_encoder_MulitSwinTransformer
from models.transformer.optimi_entry import build_optimizer, build_optimizer_rl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import pickle
import numpy as np
from shutil import copyfile
from models.grid_feature import Grid_feature_fuse_extractor
from models.transformer.dual_branch_swin_fpn_fusion import FusionModule
from models.transformer.IDMModule import IDMModule
from models.transformer import encoders
import warnings
from util import evaluate_loss, evaluate_metrics, train_xe, train_scst, predict_captions

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

warnings.filterwarnings("ignore")

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
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--val_test_batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--xe_least', type=int, default=15)
    parser.add_argument('--xe_most', type=int, default=20)
    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--refine_epoch_rl', type=int, default=28)
    parser.add_argument('--resume_last', type=bool, default=True)
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--IMG_SIZE', type=int, default=224)
    parser.add_argument('--COLOR_JITTER', type=float, default=0.4)
    parser.add_argument('--AUTO_AUGMENT', type=str, default='none', choices=['rand-m9-mstd0.5','none'])
    parser.add_argument('--REPROB', type=float, default=0.5)
    parser.add_argument('--REMODE', type=str, default='pixel')
    parser.add_argument('--RECOUNT', type=int, default=1)
    parser.add_argument('--INTERPOLATION', type=str, default='bicubic')
    parser.add_argument("--do_debug", action='store_true', help="Whether debug or not")
    args = parser.parse_args()
    print(args)

    device_a = torch.device('cuda:{}'.format(args.gpu_number[0]))
    device_b = torch.device('cuda:{}'.format(args.gpu_number[1]))
    device_c = torch.device('cuda:{}'.format(args.gpu_number[2]))
    device_d = torch.device('cuda:{}'.format(args.gpu_number[3]))

    device = torch.device(device_b)

    print('Transformer Training')
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))
    image_field = ImageField(config=args)
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    dataset = Emphasis5k(image_field, text_field, '/', args.annotation_folder, args.annotation_folder)

    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_{}.pkl'.format(args.dataset)):
        print("Rank{}: Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_{}.pkl'.format(args.dataset), 'wb'))
    else:
        print('Rank{}: Loading from vocabulary')
        text_field.vocab = pickle.load(open('vocab_{}.pkl'.format(args.dataset), 'rb'))

    image_enhancement = IDMModule().to(device_a)
    backbone_1 = build_encoder_MulitSwinTransformer(config=args).to(device_b)
    backbone_2 = build_encoder_MulitSwinTransformer(config=args).to(device_c)
    fusion_model = FusionModule(out_channels=1024, use_attention=True, use_smoothing=True).to(device_d)
    grid_feature = Grid_feature_fuse_extractor(config=args).to(device_d)
    encoder = encoders.singal_grid_transformer(N=args.Encoder_layer, padding_idx=0, d_model=512, config=args).to(device_d)
    decoder = TransformerDecoderLayer(vocab_size=len(text_field.vocab), max_len=54, N_dec=args.Decoder_layer, padding_idx=text_field.vocab.stoi['<pad>']).to(device_d)
    model = ImageEnhanceTransformer(text_field.vocab.stoi['<bos>'], image_enhancement, backbone_1, backbone_2, fusion_model, grid_feature, encoder, decoder, config=args)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = train_dataset.text()
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    def lambda_lr(s):
        print("s:", s)
        if s <= 3:
            lr = args.xe_base_lr * s / 4
        elif s <= 10:
            lr = args.xe_base_lr
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr

    def lambda_lr_rl(s):
        refine_epoch = args.refine_epoch_rl
        print("rl_s:", s)
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr

    optim = build_optimizer(model)
    scheduler = LambdaLR(optim, lambda_lr)

    optim_rl = build_optimizer_rl(model)
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    best_test_cider = 0.
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = f'{args.save_folder}/%s_last.pth' % args.exp_name
        else:
            fname = f'{args.save_folder}/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            use_rl = data['use_rl']

            if not use_rl:
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])
            else:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler'])

            print('Resuming from epoch %d, validation loss %f, best cider %f, and best_test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))
            print('patience:', data['patience'])

    print("Training starts")

    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False,
                                      persistent_workers=True)

        dataloader_val = DataLoader(val_dataset,
                                    batch_size=args.val_test_batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

        dict_dataloader_train = DataLoader(dict_dataset_train,
                                           batch_size=args.batch_size // 5,
                                           pin_memory=True,
                                           shuffle=True,
                                           drop_last=False,
                                           num_workers=args.workers,
                                           persistent_workers=True)


        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.val_test_batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.val_test_batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field, scheduler, loss_fn, e, device)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train,
                                                             text_field, scheduler_rl, e, device,config=args)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, e, device)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, e, device)
        val_cider = scores['CIDEr']
        print("Validation scores", scores)

        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, e, device)
        test_cider = scores['CIDEr']
        print("Test scores", scores)
        writer.add_scalar('data/test_cider', test_cider, e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False

        if patience == 5:
            if e < args.xe_least:
                print('special treatment, e = {}'.format(e))
                use_rl = False
                switch_to_rl = False
                patience = 0
            elif not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0

                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

                for k in range(e - 1):
                    scheduler_rl.step()
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if e == args.xe_most:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0

                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

                for k in range(e - 1):
                    scheduler_rl.step()
                print("Switching to RL")

        if switch_to_rl and not best:
            data = torch.load(f'{args.save_folder}/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, best_cider %f, and best test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
            'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }, f'{args.save_folder}/%s_last.pth' % args.exp_name)

        if best:
            copyfile(f'{args.save_folder}/%s_last.pth' % args.exp_name,
                     f'{args.save_folder}/%s_best.pth' % args.exp_name)
        if best_test:
            copyfile(f'{args.save_folder}/%s_last.pth' % args.exp_name,
                     f'{args.save_folder}/%s_best_test.pth' % args.exp_name)

        if e >= 25:
            copyfile(f'{args.save_folder}/%s_last.pth' % args.exp_name, '{}/{}_{}.pth'.format(args.save_folder, args.exp_name, e))
            pass
        if exit_train:
            writer.close()
            break



