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
from advent.utils.viz_segmask import colorize_mask
from tps.utils.resample2d_package.resample2d import Resample2d
from PIL import Image, ImageFilter

def train_domain_adaptation(model, source_loader, target_loader, cfg):
    if cfg.TRAIN.DA_METHOD == 'SourceOnly':
        train_source_only(model, source_loader, target_loader, cfg)
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
        '''
            {d, c} or {b, a}: pair of consecutive frames extracted from the same video
        '''
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
            trg_img_b_wk = torch.flip(trg_img_b, [3])
            trg_img_a_wk = torch.flip(trg_img_a, [3])
            trg_flow_b_wk = torch.flip(trg_flow_b, [3])
        else:
            trg_img_b_wk = trg_img_b
            trg_img_a_wk = trg_img_a
            trg_flow_b_wk = trg_flow_b
        # concatenate {d, c}
        trg_img_concat = torch.cat((trg_img_d, trg_img_c), 2)
        # strong augment {d, c}
        aug = T.Compose([
            T.ToPILImage(),
            T.RandomApply([GaussianBlur(radius=random.choice([5, 7, 9]))], p=0.6),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor()
        ])
        trg_img_concat_st = aug(torch.squeeze(trg_img_concat)).unsqueeze(dim=0)
        # seperate {d, c}
        trg_img_d_st = trg_img_concat_st[:, :, 0:512, :]
        trg_img_c_st = trg_img_concat_st[:, :, 512:, :]
        # rescale {d, c}
        scale_ratio = np.random.randint(100.0 * cfg.TRAIN.SCALING_RATIO[0], 100.0 * cfg.TRAIN.SCALING_RATIO[1]) / 100.0
        trg_scaled_size = (round(input_size_target[1] * scale_ratio / 8) * 8, round(input_size_target[0] * scale_ratio / 8) * 8)
        trg_interp_sc = nn.Upsample(size=trg_scaled_size, mode='bilinear', align_corners=True)
        trg_img_d_st = trg_interp_sc(trg_img_d_st)
        trg_img_c_st = trg_interp_sc(trg_img_c_st)
        ##  Temporal Pseudo Supervision  ##
        # Cross Frame Pseudo Label
        with torch.no_grad():
            trg_pred_aux, trg_pred, _, _, _, _ = model(trg_img_b_wk.cuda(device), trg_img_a_wk.cuda(device), trg_flow_b_wk, device)
            # softmax
            trg_prob = F.softmax(trg_pred, dim=1)
            trg_prob_aux = F.softmax(trg_pred_aux, dim=1)
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

def hist_match(img_src, img_trg):
    import skimage
    from skimage import exposure
    img_src = np.asarray(img_src.squeeze(0).transpose(0, 1).transpose(1, 2), np.float32)
    img_trg = np.asarray(img_trg.squeeze(0).transpose(0, 1).transpose(1, 2), np.float32)
    images_aug = exposure.match_histograms(img_src, img_trg, multichannel=True)
    return torch.from_numpy(images_aug).transpose(1, 2).transpose(0, 1).unsqueeze(0)
    
class GaussianBlur(object):

    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        
<<<<<<< HEAD
=======
class EMA(object):

    def __init__(self, model, alpha=0.999):
        """ Model exponential moving average. """
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        # NOTE: Buffer values are for things that are not parameters,
        # such as batch norm statistics
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name].copy_(
                    decay * self.shadow[name] + (1 - decay) * state[name])
        self.step += 1

    def update_buffer(self):
        # No EMA for buffer values (for now)
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name].copy_(state[name])

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


>>>>>>> df2eee379fb038dc85cfe2b0511b2465d4dbf1e4
