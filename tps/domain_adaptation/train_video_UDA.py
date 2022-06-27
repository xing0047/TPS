import os
import sys
import random
from pathlib import Path
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as T
from tqdm import tqdm
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.func import hist_match
from advent.utils.func import GaussianBlur, HorizontalFlip
from advent.utils.viz_segmask import colorize_mask
from davsn.utils.resample2d_package.resample2d import Resample2d

def train_domain_adaptation(model, source_loader, target_loader, cfg):
    if cfg.TRAIN.DA_METHOD == 'SourceOnly':
        train_source_only(model, source_loader, target_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'DAVSN':
        train_CTCR(model, source_loader, target_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'PixMatch':
        train_PixMatch(model, source_loader, target_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'TPS':
        train_TPS(model, source_loader, target_loader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")


def train_source_only(model, source_loader, target_loader, cfg):
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                                align_corners=True)
    source_loader_iter = enumerate(source_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        ######### Source-domain supervised training
        _, source_batch = source_loader_iter.__next__()
        src_img_cf, src_label, src_img_kf, src_label_kf,  _, src_img_name, _, _ = source_batch
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
        file_name = src_img_name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        src_flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf, src_pred_kf_aux, src_pred_kf = model(
            src_img_cf.cuda(device), src_img_kf.cuda(device), src_flow, device)
        src_pred = interp_source(src_pred)
        loss_seg_src_main = loss_calc(src_pred, src_label, device)

        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux, src_label, device)
        else:
            loss_seg_src_aux = 0

        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux
        loss.backward()
        
        optimizer.step()
        current_losses = {'loss_src': loss_seg_src_main,
                          'loss_src_aux': loss_seg_src_aux}
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

def train_CTCR(model, source_loader, target_loader, cfg):
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True
    # DISCRIMINATOR NETWORK
    d_sta_aux = get_fc_discriminator(num_classes=num_classes * 2)
    d_sta_aux.train()
    d_sta_aux.to(device)
    d_sta_main = get_fc_discriminator(num_classes=num_classes * 2)
    d_sta_main.train()
    d_sta_main.to(device)
    d_sa_aux = get_fc_discriminator(num_classes=num_classes * 2)
    d_sa_aux.train()
    d_sa_aux.to(device)
    d_sa_main = get_fc_discriminator(num_classes=num_classes * 2)
    d_sa_main.train()
    d_sa_main.to(device)

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # discriminators' optimizers
    optimizer_d_sta_aux = optim.Adam(d_sta_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                     betas=(0.9, 0.99))
    optimizer_d_sta_main = optim.Adam(d_sta_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                      betas=(0.9, 0.99))
    optimizer_d_sa_aux = optim.Adam(d_sa_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))
    optimizer_d_sa_main = optim.Adam(d_sa_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                     betas=(0.9, 0.99))
    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                                align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    # propagate predictions (of previous frames) forward
    warp_bilinear = Resample2d(bilinear=True)
    # labels for adversarial training
    source_label = 0
    target_label = 1
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_sta_aux.zero_grad()
        optimizer_d_sta_main.zero_grad()
        optimizer_d_sa_aux.zero_grad()
        optimizer_d_sa_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_sta_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_sta_main, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_sa_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_sa_main, i_iter, cfg)

        ######### Source-domain supervised training
        for param in d_sta_aux.parameters():
            param.requires_grad = False
        for param in d_sta_main.parameters():
            param.requires_grad = False
        for param in d_sa_aux.parameters():
            param.requires_grad = False
        for param in d_sa_main.parameters():
            param.requires_grad = False
        _, source_batch = source_loader_iter.__next__()
        src_img_cf, src_label, src_img_kf, _, _, src_img_name = source_batch
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
        file_name = src_img_name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        src_flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf, src_pred_kf_aux, src_pred_kf = model(
            src_img_cf.cuda(device), src_img_kf.cuda(device), src_flow, device)
        src_pred = interp_source(src_pred)
        loss_seg_src_main = loss_calc(src_pred, src_label, device)
        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux, src_label, device)
        else:
            loss_seg_src_aux = 0
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        ######### Usupervised domain adaptation
        _, target_batch = target_loader_iter.__next__()
        trg_img_cf, _, image_trg_kf, _, name = target_batch
        file_name = name[0].split('/')[-1]
        frame = int(file_name.replace('_leftImg8bit.png', '')[-6:])
        frame1 = frame - 1
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frame1).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        trg_pred_aux, trg_pred, trg_pred_cf_aux, trg_pred_cf, trg_pred_kf_aux, trg_pred_kf = model(
            trg_img_cf.cuda(device), image_trg_kf.cuda(device), trg_flow, device)

        ###### Cross-domain TCR
        adversarial_factor_aux = cfg.TRAIN.LAMBDA_ADV_AUX / cfg.TRAIN.LAMBDA_ADV_MAIN  # as in Advent
        ### adversarial training ot fool the discriminator
        # spatial-temporal alignment (sta)
        src_sta_pred = torch.cat((src_pred_cf, src_pred_kf), dim=1)
        trg_sta_pred = torch.cat((trg_pred_cf, trg_pred_kf), dim=1)
        src_sta_pred = interp_source(src_sta_pred)
        trg_sta_pred = interp_target(trg_sta_pred)
        d_out_sta = d_sta_main(F.softmax(trg_sta_pred))
        loss_sta = bce_loss(d_out_sta, source_label)
        if cfg.TRAIN.MULTI_LEVEL:
            src_sta_pred_aux = torch.cat((src_pred_cf_aux, src_pred_kf_aux), dim=1)
            trg_sta_pred_aux = torch.cat((trg_pred_cf_aux, trg_pred_kf_aux), dim=1)
            src_sta_pred_aux = interp_source(src_sta_pred_aux)
            trg_sta_pred_aux = interp_target(trg_sta_pred_aux)
            d_out_sta_aux = d_sta_aux(F.softmax(trg_sta_pred_aux))
            loss_sta_aux = bce_loss(d_out_sta_aux, source_label)
        else:
            loss_sta_aux = 0
        loss = (cfg.TRAIN.lamda_u * loss_sta
                       + cfg.TRAIN.lamda_u * adversarial_factor_aux * loss_sta_aux)
        # spatial alignment (sa)
        src_sa_pred = torch.cat((src_pred_cf, src_pred_cf), dim=1)
        trg_sa_pred = torch.cat((trg_pred_cf, trg_pred_cf), dim=1)
        src_sa_pred = interp_source(src_sa_pred)
        trg_sa_pred = interp_target(trg_sa_pred)
        d_out_sa = d_sa_main(F.softmax(trg_sa_pred))
        loss_sa = bce_loss(d_out_sa, source_label)
        if cfg.TRAIN.MULTI_LEVEL:
            src_sa_pred_aux = torch.cat((src_pred_cf_aux, src_pred_cf_aux), dim=1)
            trg_sa_pred_aux = torch.cat((trg_pred_cf_aux, trg_pred_cf_aux), dim=1)
            src_sa_pred_aux = interp_source(src_sa_pred_aux)
            trg_sa_pred_aux = interp_target(trg_sa_pred_aux)
            d_out_sa_aux = d_sa_aux(F.softmax(trg_sa_pred_aux))
            loss_sa_aux = bce_loss(d_out_sa_aux, source_label)
        else:
            loss_sa_aux = 0
        loss = loss + (cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_sa * loss_sa
                       + cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_sa * adversarial_factor_aux * loss_sa_aux)
        loss.backward()
        ### Train discriminator networks (Enable training mode on discriminator networks)
        for param in d_sta_aux.parameters():
            param.requires_grad = True
        for param in d_sta_main.parameters():
            param.requires_grad = True
        for param in d_sa_aux.parameters():
            param.requires_grad = True
        for param in d_sa_main.parameters():
            param.requires_grad = True
        ## Train with source
        # spatial-temporal alignment (sta)
        src_sta_pred = src_sta_pred.detach()
        d_out_sta = d_sta_main(F.softmax(src_sta_pred))
        loss_d_sta = bce_loss(d_out_sta, source_label) / 2
        loss_d_sta.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            src_sta_pred_aux = src_sta_pred_aux.detach()
            d_out_sta_aux = d_sta_aux(F.softmax(src_sta_pred_aux))
            loss_d_sta_aux = bce_loss(d_out_sta_aux, source_label) / 2
            loss_d_sta_aux.backward()
        # spatial alignment (sa)
        src_sa_pred = src_sa_pred.detach()
        d_out_sa = d_sa_main(F.softmax(src_sa_pred))
        loss_d_sa = bce_loss(d_out_sa, source_label) / 2
        loss_d_sa.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            src_sa_pred_aux = src_sa_pred_aux.detach()
            d_out_sa_aux = d_sa_aux(F.softmax(src_sa_pred_aux))
            loss_d_sa_aux = bce_loss(d_out_sa_aux, source_label) / 2
            loss_d_sa_aux.backward()
        ## Train with target
        # spatial-temporal alignment (sta)
        trg_sta_pred = trg_sta_pred.detach()
        d_out_sta = d_sta_main(F.softmax(trg_sta_pred))
        loss_d_sta = bce_loss(d_out_sta, target_label) / 2
        loss_d_sta.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            trg_sta_pred_aux = trg_sta_pred_aux.detach()
            d_out_sta_aux = d_sta_aux(F.softmax(trg_sta_pred_aux))
            loss_d_sta_aux = bce_loss(d_out_sta_aux, target_label) / 2
            loss_d_sta_aux.backward()
        else:
            loss_d_sta_aux = 0
        # spatial alignment (sa)
        trg_sa_pred = trg_sa_pred.detach()
        d_out_sa = d_sa_main(F.softmax(trg_sa_pred))
        loss_d_sa = bce_loss(d_out_sa, target_label) / 2
        loss_d_sa.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            trg_sa_pred_aux = trg_sa_pred_aux.detach()
            d_out_sa_aux = d_sa_aux(F.softmax(trg_sa_pred_aux))
            loss_d_sa_aux = bce_loss(d_out_sa_aux, target_label) / 2
            loss_d_sa_aux.backward()
        else:
            loss_d_sa_aux = 0

        # Discriminators' weights discrepancy (wd)
        k = 0
        loss_wd = 0
        for (W1, W2) in zip(d_sta_main.parameters(), d_sa_main.parameters()):
            W1 = W1.view(-1)
            W2 = W2.view(-1)
            loss_wd = loss_wd + (torch.matmul(W1, W2) / (torch.norm(W1) * torch.norm(W2)) + 1)
            k += 1
        loss_wd = loss_wd / k
        if cfg.TRAIN.MULTI_LEVEL:
            k = 0
            loss_wd_aux = 0
            for (W1_aux, W2_aux) in zip(d_sta_aux.parameters(), d_sa_aux.parameters()):
                W1_aux = W1_aux.view(-1)
                W2_aux = W2_aux.view(-1)
                loss_wd_aux = loss_wd_aux + (
                            torch.matmul(W1_aux, W2_aux) / (torch.norm(W1_aux) * torch.norm(W2_aux)) + 1)
                k += 1
            loss_wd_aux = loss_wd_aux / k
        else:
            loss_wd_aux = 0
        loss = (cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_wd * loss_wd + cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_wd * loss_wd_aux)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_sta_aux.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_sta_main.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_sa_aux.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_sa_main.parameters(), 1)
        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_sta_aux.step()
            optimizer_d_sa_aux.step()
        optimizer_d_sta_main.step()
        optimizer_d_sa_main.step()
        current_losses = {'loss_src_aux': loss_seg_src_aux,
                          'loss_src': loss_seg_src_main,
                          'loss_sta_aux': loss_sta_aux,
                          'loss_sa_aux': loss_sa_aux,
                          'loss_sta': loss_sta,
                          'loss_sa': loss_sa,
                          'loss_d_sta_aux': loss_d_sta_aux,
                          'loss_d_sa_aux': loss_d_sa_aux,
                          'loss_d_sta': loss_d_sta,
                          'loss_d_sa': loss_d_sa,
                          'loss_wd_aux': loss_wd_aux,
                          'loss_wd': loss_wd}
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

def train_PixMatch(model, source_loader, target_loader, cfg):
    
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    # propagate predictions (of previous frames) forward
    warp_bilinear = Resample2d(bilinear=True)
    #
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)): 
        
        ####  optimizer  ####
        optimizer.zero_grad()
        
        ####  adjust LR  ####
        adjust_learning_rate(optimizer, i_iter, cfg)
        
        ####  load data  ####
        _, source_batch = source_loader_iter.__next__()
        src_img_cf, src_label, src_img_kf, src_label_kf, _, src_img_name, src_cf, src_kf = source_batch

        _, target_batch = target_loader_iter.__next__()
        trg_img_d, trg_img_c, _, _, _,  _, name, frames = target_batch 
        frames = frames.squeeze().tolist()

        ####  supervised | source domain  ####
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
        file_name = src_img_name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        src_flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf, src_pred_kf_aux, src_pred_kf = model(src_img_cf.cuda(device), src_img_kf.cuda(device), src_flow, device)
        src_pred = interp_source(src_pred)
        loss_seg_src_main = loss_calc(src_pred, src_label, device)
        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux, src_label, device)
        else:
            loss_seg_src_aux = 0
        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux
        loss.backward()

        ####  unsupervised | target domain  ####
        ##  optical flow  ##
        file_name = name[0].split('/')[-1]
        # flow: d -> c
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[1]).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow_d = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        ##  augmentation  ##
        # weak #
        # flip {d, c}
        flip = random.random() < 0.5
        if flip:
            trg_img_d_wk = torch.flip(trg_img_d, [3])
            trg_img_c_wk = torch.flip(trg_img_c, [3])
            trg_flow_d_wk = torch.flip(trg_flow_d, [3])
        # strong #
        # concatenate {d, c}
        trg_img_concat = torch.cat((trg_img_d, trg_img_c), 2)
        # strong augment {d, c}
        aug = T.Compose([
            T.ToPILImage(),
            T.RandomApply([GaussianBlur(radius=random.choice([7, 9, 11]))], p=0.6),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor()
        ])
        trg_img_concat_st = aug(torch.squeeze(trg_img_concat)).unsqueeze(dim=0)
        # seperate {d, c}
        trg_img_d_st = trg_img_concat_st[:, :, 0:512, :]
        trg_img_c_st = trg_img_concat_st[:, :, 512:, :]
        # rescale {d, c}
        scale_ratio = np.random.randint(100.0*cfg.TRAIN.SCALING_RATIO[0], 100.0*cfg.TRAIN.SCALING_RATIO[1])/100.0
        trg_scaled_size = (round(input_size_target[1] * scale_ratio / 8) * 8, round(input_size_target[0] * scale_ratio / 8) * 8)
        trg_interp_sc = nn.Upsample(size=trg_scaled_size, mode='bilinear', align_corners=True)
        trg_img_d_st = trg_interp_sc(trg_img_d_st)
        trg_img_c_st = trg_interp_sc(trg_img_c_st)
        
        with torch.no_grad():
            trg_pred_aux, trg_pred, _, _, _, _ = model(trg_img_d_wk.cuda(device), trg_img_c_wk.cuda(device), trg_flow_d_wk, device)
            # softmax
            trg_prob = F.softmax(trg_pred)
            trg_prob_aux = F.softmax(trg_pred_aux)
            # pseudo label
            trg_pl = torch.argmax(trg_prob, 1)
            trg_pl_aux = torch.argmax(trg_prob_aux, 1)
            if flip:
                trg_pl = torch.flip(trg_pl, [2])
                trg_pl_aux = torch.flip(trg_pl_aux, [2])
            # rescale param
            trg_interp_sc2ori = nn.Upsample(size=(trg_pred.shape[-2], trg_pred.shape[-1]), mode='bilinear', align_corners=True)
        # forward prop
        trg_pred_aux, trg_pred, _, _, _, _ = model(trg_img_d_st.cuda(device), trg_img_c_st.cuda(device), trg_flow_d, device)
        # rescale
        trg_pred = trg_interp_sc2ori(trg_pred)
        trg_pred_aux = trg_interp_sc2ori(trg_pred_aux)
        # unsupervised loss
        loss_trg = loss_calc(trg_pred, trg_pl, device)
        if cfg.TRAIN.MULTI_LEVEL:
            loss_trg_aux = loss_calc(trg_pred_aux, trg_pl_aux, device)
        else:
            loss_trg_aux = 0
        loss = cfg.TRAIN.LAMBDA_T * (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_trg + cfg.TRAIN.LAMBDA_SEG_AUX * loss_trg_aux)
        loss.backward()

        ####  step  ####
        optimizer.step()
        
        ####  logging  ####
        if cfg.TRAIN.MULTI_LEVEL:
            current_losses = {'loss_src': loss_seg_src_main,
                              'loss_src_aux': loss_seg_src_aux,
                              'loss_trg': loss_trg,
                              'loss_trg_aux': loss_trg_aux
                             }
        else:
            current_losses = {'loss_src': loss_seg_src_main,
                              'loss_trg': loss_trg
                             }
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)


def train_TPS(model, source_loader, target_loader, cfg):
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    # propagate predictions (of previous frames) forward
    warp_bilinear = Resample2d(bilinear=True)
    #
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)): 
        
        ####  optimizer  ####
        optimizer.zero_grad()
        
        ####  adjust LR  ####
        adjust_learning_rate(optimizer, i_iter, cfg)
        
        ####  load data  ####
        _, source_batch = source_loader_iter.__next__()
        src_img_cf, src_label, src_img_kf, src_label_kf, _, src_img_name, src_cf, src_kf = source_batch

        _, target_batch = target_loader_iter.__next__()
        trg_img_d, trg_img_c, trg_img_b, trg_img_a, d,  _, name, frames = target_batch 
        frames = frames.squeeze().tolist()
        ##  match
        src_cf = hist_match(src_cf, d)
        src_kf = hist_match(src_kf, d)
        ##  normalize
        src_cf = torch.flip(src_cf, [1])
        src_kf = torch.flip(src_kf, [1])
        src_cf -= torch.tensor(cfg.TRAIN.IMG_MEAN).view(1, 3, 1, 1)
        src_kf -= torch.tensor(cfg.TRAIN.IMG_MEAN).view(1, 3, 1, 1)
        ##  recover
        src_img_cf = src_cf
        src_img_kf = src_kf

        ####  supervised | source  ####
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
        file_name = src_img_name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        src_flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf, src_pred_kf_aux, src_pred_kf = model(src_img_cf.cuda(device), src_img_kf.cuda(device), src_flow, device)
        src_pred = interp_source(src_pred)
        loss_seg_src_main = loss_calc(src_pred, src_label, device)
        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux, src_label, device)
        else:
            loss_seg_src_aux = 0
        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux
        loss.backward()

        ####  unsupervised | target  ####
        ##  optical flow  ##
        file_name = name[0].split('/')[-1]
        # flow: d -> c
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[1]).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow_d = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        # flow: d -> b
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[2]).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        # flow: b -> a
        file_name = file_name.replace(str(frames[0]).zfill(6), str(frames[2]).zfill(6))
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[3]).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow_b = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        ##  augmentation  ##
        # flip {b, a}
        flip = random.random() < 0.5
        if flip:
            trg_img_b = torch.flip(trg_img_b, [3])
            trg_img_a = torch.flip(trg_img_a, [3])
            trg_flow_b = torch.flip(trg_flow_b, [3])
        # concatenate {d, c}
        trg_img_concat = torch.cat((trg_img_d, trg_img_c), 2)
        # strong augment {d, c}
        aug = T.Compose([
            T.ToPILImage(),
            T.RandomApply([GaussianBlur(radius=random.choice([7, 9, 11]))], p=0.6),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor()
        ])
        trg_img_concat_st = aug(torch.squeeze(trg_img_concat)).unsqueeze(dim=0)
        # seperate {d, c}
        trg_img_d = trg_img_concat_st[:, :, 0:512, :]
        trg_img_c = trg_img_concat_st[:, :, 512:, :]
        # rescale {d, c}
        scale_ratio = np.random.randint(100.0*cfg.TRAIN.SCALING_RATIO[0], 100.0*cfg.TRAIN.SCALING_RATIO[1])/100.0
        trg_scaled_size = (round(input_size_target[1] * scale_ratio / 8) * 8, round(input_size_target[0] * scale_ratio / 8) * 8)
        trg_interp_sc = nn.Upsample(size=trg_scaled_size, mode='bilinear', align_corners=True)
        trg_img_d = trg_interp_sc(trg_img_d)
        trg_img_c = trg_interp_sc(trg_img_c)
        ##  Temporal Pseudo Supervision  ##
        # Cross Frame Pseudo Label
        with torch.no_grad():
            trg_pred_aux, trg_pred, _, _, _, _ = model(trg_img_b.cuda(device), trg_img_a.cuda(device), trg_flow_b, device)
            # softmax
            trg_prob = F.softmax(trg_pred)
            trg_prob_aux = F.softmax(trg_pred_aux)
            # warp
            interp_flow = nn.Upsample(size=(trg_prob.shape[-2], trg_prob.shape[-1]), mode='bilinear', align_corners=True)
            interp_flow_ratio = trg_prob.shape[-2] / trg_flow.shape[-2]
            trg_flow_warp = (interp_flow(trg_flow) * interp_flow_ratio).float().cuda(device)
            trg_prob_warp = warp_bilinear(trg_prob, trg_flow_warp)
            trg_prob_warp_aux = warp_bilinear(trg_prob_aux, trg_flow_warp)
            # pseudo label
            trg_pl = torch.argmax(trg_prob_warp, 1)
            trg_pl_aux = torch.argmax(trg_prob_warp_aux, 1)
            if flip:
                trg_pl = torch.flip(trg_pl, [2])
                trg_pl_aux = torch.flip(trg_pl_aux, [2])
            # rescale param
            trg_interp_sc2ori = nn.Upsample(size=(trg_pred.shape[-2], trg_pred.shape[-1]), mode='bilinear', align_corners=True)
        # forward prop
        trg_pred_aux, trg_pred, _, _, _, _ = model(trg_img_d.cuda(device), trg_img_c.cuda(device), trg_flow_d, device)
        # rescale
        trg_pred = trg_interp_sc2ori(trg_pred)
        trg_pred_aux = trg_interp_sc2ori(trg_pred_aux)
        # unsupervised loss
        loss_trg = loss_calc(trg_pred, trg_pl, device)
        if cfg.TRAIN.MULTI_LEVEL:
            loss_trg_aux = loss_calc(trg_pred_aux, trg_pl_aux, device)
        else:
            loss_trg_aux = 0
        loss = cfg.TRAIN.LAMBDA_T * (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_trg + cfg.TRAIN.LAMBDA_SEG_AUX * loss_trg_aux)
        loss.backward()

        ####  step  ####
        optimizer.step()
        
        ####  logging  ####
        if cfg.TRAIN.MULTI_LEVEL:
            current_losses = {'loss_src': loss_seg_src_main,
                              'loss_src_aux': loss_seg_src_aux,
                              'loss_trg': loss_trg,
                              'loss_trg_aux': loss_trg_aux
                             }
        else:
            current_losses = {'loss_src': loss_seg_src_main,
                              'loss_trg': loss_trg
                             }
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

## utils
def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

