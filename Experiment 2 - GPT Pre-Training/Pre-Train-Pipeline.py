import torch
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
                                # Model:
                                tokens_batch_size = 512 * 40 * 1, #No. of tokens in one iteration of gradient accumulation
                                batch_size = 40,
                                dec_context_size = 512,
                                batch_overlap = 0, 
                                betas = (0.9, 0.95),
                                vocab_size = 50259, 
                                d_model = 512,
                                num_heads = 8,
                                num_decoder_blocks = 8,
                                pos_enc_dropout = 0,
                                drop_prob = 0,
                                weight_decay = 0,
                                d_feedfwd = 512 * 4,
                                mask_attention = True,
                                pre_norm = True,

                                # Data Loader and Checkpointing:
                                x_data_loader_dtype = torch.int32,
                                y_data_loader_dtype = torch.int64,
                                load_check_point = False,
                                checkpoint_path = './CheckPoints/Fine-Web-Edu-Val/',
                                checkpoint_name = '',
                                checkpoint_save_iter = 6000,
                                num_iters = 48005, #No. of iterations for training 10 Shards (~ 1 Billion Tokens)
                                eval_val_set = True,
                                val_eval_iters = 50,
                                val_eval_interval = 200,
                                
                                # Optimization:
                                optimizer_name = None,
                                max_lr = None,
                                min_lr = None,
                                model_name = 'CFGs-Fine-Web-Edu-',
                                start_lr = None,
                                lr_decrement = None,
                                lr_increment = None,
                                warmup_iters = None,
                                clip_grad_norm = False,

                                # Training Files:
                                replacements = {},
                                file_name = "Fine-Web-Edu",
                                file_path = None,
                                vocab_path = "GPT-2",
                                load_merge_info_name = 'GPT-2',
                                load_vocab_name = 'GPT-2',
                                data_path = './Datasets/FineWeb-Edu/',
                                train_shard_names =    [
                                                    'Fine-Web-Edu-Sample-1BT-Shard-1',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-2',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-3',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-4',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-5',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-6',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-7',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-8',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-9',
                                                    'Fine-Web-Edu-Sample-1BT-Shard-10'
                                                ],
                                val_name = 'Fine-Web-Edu-Sample-1BT-Shard-0',
                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            ))

config_nos = config_nos

pipe_indx = 0
cfg_no = -1

cfg_no += 1
configs[cfg_no].optimizer_name = PLAIN_CUSTOM_ADAM
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-Shards-CFG-' + str(cfg_no) + '-Fine-Web-Edu'
configs[cfg_no].start_lr = 7.5e-5
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = CUSTOM_ADAM
configs[cfg_no].max_lr = 7.5e-5
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-Shards-CFG-' + str(cfg_no) + '-Fine-Web-Edu'
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = PLAIN_CUSTOM_ADAM
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-Shards-CFG-' + str(cfg_no) + '-Fine-Web-Edu'
configs[cfg_no].start_lr = 10e-4
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = CUSTOM_ADAM
configs[cfg_no].max_lr = 10e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-Shards-CFG-' + str(cfg_no) + '-Fine-Web-Edu'
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = ACC_ASRO_FINAL_SCALE
configs[cfg_no].max_lr = 10e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-Shards-CFG-' + str(cfg_no) + '-Fine-Web-Edu'
configs[cfg_no].start_lr = 7.5e-5
configs[cfg_no].lr_decrement = 2e-4
configs[cfg_no].lr_increment = 5e-2
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = R_ADAM
configs[cfg_no].max_lr = 1e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-Shards-CFG-' + str(cfg_no) + '-Fine-Web-Edu'
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = AMS_GRAD
configs[cfg_no].max_lr = 1e-4
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-Shards-CFG-' + str(cfg_no) + '-Fine-Web-Edu'
configs[cfg_no].warmup_iters = 2

cfg_no += 1
configs[cfg_no].optimizer_name = ASRO
configs[cfg_no].min_lr = 1e-5
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-Shards-CFG-' + str(cfg_no) + '-Fine-Web-Edu'
configs[cfg_no].start_lr = 10e-4
configs[cfg_no].lr_decrement = 2e-4
configs[cfg_no].warmup_iters = 2

for i in range(config_nos):
    trainer = Trainer(configs[i])
    trainer.train()
    del trainer
