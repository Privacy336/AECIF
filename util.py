import torch
from tqdm import tqdm
import evaluation
import itertools
import numpy as np


def evaluate_loss(model, dataloader, loss_fn, text_field, e, device):
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (iamge, image_id, captions) in enumerate(dataloader):
                images, captions = iamge.to(device), captions.to(device)
                detections = (images, image_id)
                out = model(mode = 'xe', img_id = detections, seq = captions).to(device)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, e, device):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc = 'Epoch %d - evaluation' % e, unit = 'it', total = len(dataloader)) as pbar:
        for it, (images_list, caps_gt) in enumerate(iter(dataloader)):
            images = images_list[0].to(device)
            images_id_list = (images, images_list[1])
            with torch.no_grad():
                out, _ = model(mode = 'rl',
                               img_id = images_id_list,
                               max_len = 20,
                               eos_idx = text_field.vocab.stoi['<eos>'],
                               beam_size = 5,
                               out_size = 1)
                out = out.to(device)
            caps_gen = text_field.decode(out, join_words = False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def predict_captions(model, dataloader, text_field, device):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images_list, caps_gt) in enumerate(iter(dataloader)):
            images = images_list[0].to(device)
            images_id_list = (images, images_list[1])
            with torch.no_grad():

                out, _ = model(mode='rl',
                               img_id=images_id_list,
                               max_len=20,
                               eos_idx=text_field.vocab.stoi['<eos>'],
                               beam_size=5,
                               out_size=1)

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def train_xe(model, dataloader, optim, text_field, scheduler, loss_fn, e, device):
    model.train()
    scheduler.step()
    print('Backbone lr = ', optim.state_dict()['param_groups'][0]['lr'])
    print('Dec lr = ', optim.state_dict()['param_groups'][1]['lr'])
    running_loss = .0
    with tqdm(desc = 'Epoch %d - train' % e, unit = 'it', total = len(dataloader)) as pbar:
        for it, (iamge, image_id,captions) in enumerate(dataloader):
            images, captions = iamge.to(device), captions.to(device)
            detections = (images, image_id)
            out = model(mode = 'xe', img_id = detections, seq = captions).to(device)

            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss +=  this_loss

            pbar.set_postfix(loss = running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss
def train_scst(model, dataloader, optim_rl, cider, text_field, scheduler_rl, e, device,config):
    running_reward = .0
    running_reward_baseline = .0

    model.train()
    scheduler_rl.step()
    if device ==  0:
        print('lr = ', optim_rl.state_dict()['param_groups'][0]['lr'])
    running_loss = .0
    seq_len = 20
    beam_size = 5
    with tqdm(desc = 'Epoch %d - train' % e, unit = 'it', total = len(dataloader)) as pbar:
        for it, (images_list, caps_gt) in enumerate(iter(dataloader)):
            images = images_list[0].to(device)
            detections = (images, images_list[1])
            outs, log_probs = model(mode = 'rl',
                                    img_id = detections,
                                    max_len = seq_len,
                                    eos_idx = text_field.vocab.stoi['<eos>'],
                                    beam_size = beam_size,
                                    out_size = beam_size)

            optim_rl.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen = evaluation.PTBTokenizer.tokenize(caps_gen)
            caps_gt = evaluation.PTBTokenizer.tokenize(caps_gt)
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections[0].shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim = True)



            log_probs = log_probs.to('cuda:{}'.format(config.gpu_number[1]))

            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)



            loss = loss.mean()
            loss.backward()
            optim_rl.step()

            running_loss +=  loss.item()
            running_reward +=  reward.mean().item()
            running_reward_baseline +=  reward_baseline.mean().item()
            pbar.set_postfix(loss = running_loss / (it + 1), reward = running_reward / (it + 1),
                             reward_baseline = running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline



