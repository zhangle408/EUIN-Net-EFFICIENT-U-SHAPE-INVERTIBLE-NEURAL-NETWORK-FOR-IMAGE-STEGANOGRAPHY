import warnings
import sys
import math
import os
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
import tqdm
# import cv2
from model import *
# from imp_subnet import *
import config as c
from os.path import join
import datasets
import modules.module_util as mutil
import modules.Unet_common as common
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0,1]



def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def bgr2ycbcr(img, only_y=False, only_u=False,only_v=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    # if only_y:
    #     rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    # if only_u:
    #     rlt = np.dot(img, [128.553, -74.203, -93.786]) / 255.0 + 128
    if only_v:
        rlt = np.dot(img, [65.481, -37.797, 112.0]) / 255.0 + 128
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


net1 = Model()
net1.cuda()
init_model(net1)
net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
dwt = common.DWT()
iwt = common.IWT()

if c.pretrain:
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '.pt', net1, optim1)



with torch.no_grad():
    net1.eval()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    psnr_C = AverageMeter()
    psnr_S = AverageMeter()
    ssim_C = AverageMeter()
    ssim_S = AverageMeter()
    lpips_C = AverageMeter()
    lpips_S = AverageMeter()

    import time
    start = time.time()
    for i, xx in enumerate(datasets.testloader):
        data, lable = xx
        data = data.to(device)

        secret_1 = data[data.shape[0] // 3:]  # data
        cover = data[:data.shape[0] // 3]  # data
        secret_1 = secret_1.view(secret_1.shape[0] // 2, secret_1.shape[1] * 2, secret_1.shape[2], secret_1.shape[3])

        cover_dwt = dwt(cover)  # channels = 12
        secret_dwt_1 = dwt(secret_1)

        input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24


        #################
        #    forward1:   #
        #################
        output_dwt_1 = net1(input_dwt_1)  # channels = 24
        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, output_dwt_1.shape[1] - 4 * c.channels_in)  # channels = 12

        # get steg1
        output_steg_1 = iwt(output_steg_dwt_1).to(device)  # channels = 3

        output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
        output_rev_dwt_1 = torch.cat((output_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 36

        rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, output_dwt_1.shape[1] - 4 * c.channels_in)  # channels = 12
        rev_secret_1 = iwt(rev_secret_dwt).to(device)
        rev_secret_img = rev_secret_1
        cover_imgv = cover
        container_img = output_steg_1
        secret_imgv_nh = secret_1
        diffH = (container_img - cover_imgv).abs().mean() * 255
        diffR = (rev_secret_img - secret_imgv_nh).abs().mean() * 255
        N, _, _, _ = rev_secret_img.shape

        cover_img_numpy = cover_imgv.clone().cpu().detach().numpy()
        container_img_numpy = container_img.clone().cpu().detach().numpy()

        cover_img_numpy = cover_img_numpy.transpose(0, 2, 3, 1)
        container_img_numpy = container_img_numpy.transpose(0, 2, 3, 1)

        rev_secret_numpy = rev_secret_img.cpu().detach().numpy()
        secret_img_numpy = secret_imgv_nh.cpu().detach().numpy()

        rev_secret_numpy = rev_secret_numpy.transpose(0, 2, 3, 1)
        secret_img_numpy = secret_img_numpy.transpose(0, 2, 3, 1)

        # PSNR
        print("Cover Agnostic")

        print("Secret APD C:", diffH.item())


        psnr_c = np.zeros((N, 3))
        for i in range(N):
            psnr_c[i, 0] = PSNR(cover_img_numpy[i, :, :, 0], container_img_numpy[i, :, :, 0])
            psnr_c[i, 1] = PSNR(cover_img_numpy[i, :, :, 1], container_img_numpy[i, :, :, 1])
            psnr_c[i, 2] = PSNR(cover_img_numpy[i, :, :, 2], container_img_numpy[i, :, :, 2])
        print("Avg. PSNR C:", psnr_c.mean().item())

        # SSIM
        ssim_c = np.zeros(N)
        for i in range(N):
            ssim_c[i] = SSIM(cover_img_numpy[i], container_img_numpy[i], multichannel=True)
        print("Avg. SSIM C:", ssim_c.mean().item())

        print("Secret APD S:", diffR.item())

        psnr_s = np.zeros((N, 3))
        for i in range(N):
            psnr_s[i, 0] = PSNR(secret_img_numpy[i, :, :, 0], rev_secret_numpy[i, :, :, 0])
            psnr_s[i, 1] = PSNR(secret_img_numpy[i, :, :, 1], rev_secret_numpy[i, :, :, 1])
            psnr_s[i, 2] = PSNR(secret_img_numpy[i, :, :, 2], rev_secret_numpy[i, :, :, 2])
        print("Avg. PSNR S:", psnr_s.mean().item())


        # SSIM
        ssim_s = np.zeros(N)
        for i in range(N):
            ssim_s[i] = SSIM(secret_img_numpy[i], rev_secret_numpy[i], multichannel=True)
        print("Avg. SSIM S:", ssim_s.mean().item())


        psnr_S.update(psnr_s.mean().item(), 44 * 1 * 1)  # R loss
        ssim_S.update(ssim_s.mean().item(), 44 * 1 * 1)
        Rdiff.update(diffR.item(), 44 * 1 * 1)
        psnr_C.update(psnr_c.mean().item(), 44 * 1 * 1)  # R loss
        ssim_C.update(ssim_c.mean().item(), 44 * 1 * 1)
        Hdiff.update(diffH.item(), 44 * 1 * 1)

        if i == 200 - 1:
            break
    print('Hdiff.avg, Rdiff.avg', Hdiff.avg, Rdiff.avg)
    print('Hdiff.avg', Hdiff.avg, 'psnr_c.avg', psnr_C.avg, 'ssim_c.avg', ssim_C.avg)
    print('Rdiff.avg', Rdiff.avg, 'psnr_s.avg', psnr_S.avg, 'ssim_s.avg', ssim_S.avg)

