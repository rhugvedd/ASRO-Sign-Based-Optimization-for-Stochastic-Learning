import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b1
import gc
import time
import datetime

from Optimizers import * 

from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

from TrainConfig import *
from Trainer import *

CUSTOM_ADAM = 'CustomAdam'
PLAIN_CUSTOM_ADAM = 'PlainCustomAdam'

ASRO = 'Asro'
ACC_ASRO_FINAL_SCALE = 'AccAsroFinalScale'

RMS_PROP = 'RmsProp'
ADA_GRAD = 'AdaGrad'
AMS_GRAD = 'AmsGrad'
SGD_NESTEROV = 'SGDNesterov'
MOMENTUM = 'Momentum'
R_ADAM = 'RAdam'

configs = []

config_nos = 8

for i in range(config_nos):
    configs.append(TrainConfig(
                                batch_size = 256,
                                max_norm = False,
                                gradient_accum_iters = 1,
                                print_stat_itrvl = None,
                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                dropout = 0.2,
                                weight_decay = 0, # 1e-5
                                load_check_point = False,
                                checkpoint_path = './CheckPoints/',
                                checkpoint_name = '',
                                checkpoint_save_epoch = 10,
                                num_epochs = 61,
                                eval_val_set = True,
                                val_eval_iters = None,
                                val_eval_interval = None,
                                optimizer_name = None,
                                betas = (0.9, 0.999),
                                max_lr = None,
                                min_lr = None,
                                model_name = None,
                                start_lr = None,
                                lr_decrement = None,
                                lr_increment = None,
                                warmup_iters = None,
                                data_path = './Data/',
                                alpha = None,
                                momentum = None
    ))

config_nos = config_nos

pipe_indx = 0
cfg_no = -1

cfg_no += 1
configs[cfg_no].optimizer_name = PLAIN_CUSTOM_ADAM
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Imagenet'
configs[cfg_no].start_lr = 10e-4
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = CUSTOM_ADAM
configs[cfg_no].max_lr = 10e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Imagenet'
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = PLAIN_CUSTOM_ADAM
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Imagenet'
configs[cfg_no].start_lr = 2.5e-4
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = CUSTOM_ADAM
configs[cfg_no].max_lr = 2.5e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Imagenet'
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = ACC_ASRO_FINAL_SCALE
configs[cfg_no].max_lr = 10e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Imagenet'
configs[cfg_no].start_lr = 2.5e-4
configs[cfg_no].lr_decrement = 7.5e-5
configs[cfg_no].lr_increment = 2e-2
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = ASRO
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Imagenet'
configs[cfg_no].start_lr = 10e-4
configs[cfg_no].lr_decrement = 7.5e-5
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = AMS_GRAD
configs[cfg_no].max_lr = 10e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Imagenet'
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = R_ADAM
configs[cfg_no].max_lr = 10e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Imagenet'
configs[cfg_no].warmup_iters = 2

for i in range(config_nos):
    trainer = Trainer(configs[i])
    trainer.train()
    del trainer
    gc.collect()