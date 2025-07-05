# coding=utf-8
import argparse
import random
import os
from model import WiFo_model, WiFo
from test import Tester

import setproctitle
import torch

from DataLoader import  data_load_vision_aided_cpf
from utils import *

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from scheduler import FakeLR

def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")

def create_argparser():
    defaults = dict(
        # experimental settings
        note = 'Inference',
        task = 'Vision-aided-CPF',
        file_load_path ='./dataset',
        dataset = 'SynthSoM',
        used_data = '',
        process_name = 'process_name_inference',
        his_len = 6,
        pred_len = 6,
        few_ratio = 0.0,
        stage = 0,

        # model settings
        mask_ratio = 0.5,
        patch_size = 4,
        t_patch_size = 4,
        size = 'base',
        no_qkv_bias = 0,
        pos_emb = 'SinCos_3D',
        conv_num = 3,

        # pretrain settings
        random=True,
        mask_strategy = 'fre',
        mask_strategy_random = 'none', # ['none','batch']
        
        # training parameters
        lr=5e-6,
        min_lr = 1e-6,
        epochs = 150,
        early_stop = 5,
        weight_decay=0.000001,
        batch_size=64,
        log_interval=5,
        total_epoches = 10000,
        device_id='3',
        machine = 'machine_name',
        clip_grad = 0.05,  # 0.05
        lr_anneal_steps = 200,
        rgb_aided=True,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')


Test_Dict = {
    'Zero-shot': './Weights/wifo_base.pkl',
    'WiFo-RGB': './Weights/WiFo-RGB.pkl',
    'WiFo': './Weights/WiFo.pkl',
    'Task-Specific-RGB': './Weights/SM-RGB.pkl',
    'Task-Specific': './Weights/SM.pkl',
}

def main():

    th.autograd.set_detect_anomaly(True)
    
    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))
    setup_init(100)  # 随机种子设定100

    test_data = data_load_vision_aided_cpf(args)  # load data

    device = dev(args.device_id)
    writer = SummaryWriter(log_dir='../', flush_secs=5)

    for model_name, weight_path in Test_Dict.items():
        if 'RGB' in model_name:
            args.rgb_aided = True
        else:
            args.rgb_aided = False

        model = WiFo(t_patch_size=4, patch_size=4, embed_dim=512, decoder_embed_dim=512,
                     depth=6, decoder_depth=4, num_heads=8, decoder_num_heads=8,
                     pos_emb='SinCos_3D', args=args).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        scheduler = FakeLR(optimizer=optimizer)

        nmse = Tester(
            args=args,
            writer=writer,
            model=model,
            optimizer=optimizer, scheduler=scheduler,
            test_data=test_data,
            device=device,
            early_stop=args.early_stop,
        ).Test_iter(test_data, mask_ratio=0.5, mask_strategy='fre', seed=100,
                                               dataset='SynthSoM')
        print(model_name, nmse, 10*np.log10(nmse))


if __name__ == "__main__":
    main()