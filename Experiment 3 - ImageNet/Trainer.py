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

class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.model = efficientnet_b1(weights=None).to(config.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.all_train_losses = []
        self.train_losses = []

        self.all_train_accuracies = []
        self.train_accuracies = []

        self.val_losses = []
        self.val_accuracies = []

    @torch.no_grad()
    def estimate_val_loss(self):
        self.model.eval()
        val_loss_epoch = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            val_loss = loss.item()
            val_loss_epoch += val_loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss_epoch / len(self.val_loader)
        accuracy = 100. * correct / total
        
        self.model.train()

        return avg_val_loss, accuracy

    def train(self):
        print("Initializing")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Iterations for Gradient Accumulation: {self.config.gradient_accum_iters}")

        torch.cuda.empty_cache()
        
        print(f"No. of Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6} M parameters\n")
        
        #====================================================================================================================
        
        print("Preparing Data Loaders")

        train_transforms = transforms.Compose([
            transforms.Resize(208),
            transforms.CenterCrop(208),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transforms = train_transforms

        train_dataset = ImageNet(self.config.data_path, split='train', transform = train_transforms)
        val_dataset = ImageNet(self.config.data_path, split='val', transform = val_transforms)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=5, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=5, pin_memory=True)

        print("Preparing Complete\n")

        #====================================================================================================================

        print("Configuring Optimzer and Learning Rate Scheduler")
        
        tot_iters = (self.config.num_epochs * len(train_loader) // self.config.gradient_accum_iters) + 10
        iters = torch.arange(tot_iters + 1)

        if (self.config.optimizer_name == ACC_ASRO_FINAL_SCALE) or (self.config.optimizer_name == ASRO) or (self.config.optimizer_name == PLAIN_CUSTOM_ADAM):
            lr_schedule = (torch.ones(tot_iters + 1) * self.config.start_lr)
            lr_schedule[:self.config.warmup_iters] = self.config.start_lr * torch.arange(self.config.warmup_iters) / self.config.warmup_iters
        elif (self.config.optimizer_name == CUSTOM_ADAM) or (self.config.optimizer_name == RMS_PROP) or (self.config.optimizer_name == ADA_GRAD) or (self.config.optimizer_name == AMS_GRAD) or (self.config.optimizer_name == SGD_NESTEROV) or (self.config.optimizer_name == MOMENTUM) or (self.config.optimizer_name == R_ADAM):
            lr_schedule = (self.config.max_lr * ((iters < self.config.warmup_iters) * (iters + 1) / self.config.warmup_iters)) + ((self.config.min_lr + (0.5 * (self.config.max_lr - self.config.min_lr) * (1 + torch.cos((iters - self.config.warmup_iters) * torch.pi / (tot_iters - self.config.warmup_iters))))) * (iters >= self.config.warmup_iters))
        else:
            raise ValueError('Wrong Optimizer Name..!')

        decay_params = [param for param in self.model.parameters() if (param.requires_grad and (param.dim() >= 2))]
        no_decay_params = [param for param in self.model.parameters() if (param.requires_grad and (param.dim() < 2))]

        print(f"Number of parameter tensors with weight decay: {len(decay_params)}, totaling {sum(p.numel() for p in decay_params):,} parameters")
        print(f"Number of parameter tensors without weight decay: {len(no_decay_params)}, totaling {sum(p.numel() for p in no_decay_params):,} parameters")

        if self.config.optimizer_name == PLAIN_CUSTOM_ADAM:
            print('\nPlain Custom Adam\n')
            print(f'LR: {self.config.start_lr}\n')
            optimizer = CustomAdam (   
                                        [
                                            {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                            {'params': no_decay_params, 'weight_decay': 0.0}
                                        ],
                                        lr = lr_schedule[0],
                                        betas = self.config.betas,
                                        eps = 1e-8
                                    )
        elif self.config.optimizer_name == CUSTOM_ADAM:
            print('\nUsing Custom Adam\n')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}\n')
            optimizer = CustomAdam (   
                                        [
                                            {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                            {'params': no_decay_params, 'weight_decay': 0.0}
                                        ],
                                        lr = lr_schedule[0],
                                        betas = self.config.betas,
                                        eps = 1e-8
                                    )
        elif self.config.optimizer_name == ASRO:
            print('\nUsing Asro')
            print(f'Start(Max / Warmup Max) LR: {self.config.start_lr}')
            print(f'Decrement: {self.config.lr_decrement}')
            print(f'Min LR: {self.config.min_lr}')
            print(f'Min LR Scale Clamp: {self.config.min_lr / self.config.start_lr}')
            print(f'Decr Start Step: {self.config.warmup_iters}\n')
            optimizer = Asro(   
                                [
                                    {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                    {'params': no_decay_params, 'weight_decay': 0.0}
                                ],
                                lr = lr_schedule[0],
                                decrement = self.config.lr_decrement,
                                min_lr_scale_clamp = self.config.min_lr / self.config.start_lr,
                                decr_start_step = self.config.warmup_iters,
                                betas = self.config.betas,
                                eps = 1e-8
                            )
        elif self.config.optimizer_name == ACC_ASRO_FINAL_SCALE:
            print('\nUsing AccAsroFinalScale')
            print(f'Start(Max / Warmup Max) LR: {self.config.start_lr}')
            print(f'Increment: {self.config.lr_increment}')
            print(f'Decrement: {self.config.lr_decrement}')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}')
            print(f'Max LR Scale Clamp: {self.config.max_lr / self.config.start_lr}')
            print(f'Min LR Scale Clamp: {self.config.min_lr / self.config.start_lr}')
            print(f'Decr Start Step: {self.config.warmup_iters}\n')

            optimizer = AccAsroFinalScale (
                                    [
                                        {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                        {'params': no_decay_params, 'weight_decay': 0.0}
                                    ],
                                    lr = lr_schedule[0],
                                    increment = self.config.lr_increment,
                                    decrement = self.config.lr_decrement,
                                    max_lr_scale_clamp = self.config.max_lr / self.config.start_lr,
                                    min_lr_scale_clamp = self.config.min_lr / self.config.start_lr,
                                    decr_start_step = self.config.warmup_iters,
                                    num_iters = (self.config.num_epochs * len(train_loader) // self.config.gradient_accum_iters) + 10,
                                    betas = self.config.betas,
                                    eps = 1e-8
                                )
        elif self.config.optimizer_name == RMS_PROP:
            print('\nUsing RMS Prop')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}\n')
            print(f'Alpha: {self.config.alpha}')
            optimizer = torch.optim.RMSprop(
                                    [
                                        {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                        {'params': no_decay_params, 'weight_decay': 0.0}
                                    ],
                                    lr=lr_schedule[0],
                                    alpha=self.config.alpha,  
                                    eps=1e-8
                                )
        elif self.config.optimizer_name == MOMENTUM:
            print('\nUsing SGD - Momentum')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}\n')
            print(f'Momentum: {self.config.momentum}')
            optimizer = torch.optim.SGD ( 
                                            [
                                                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                                {'params': no_decay_params, 'weight_decay': 0.0}
                                            ],
                                            lr=lr_schedule[0],
                                            momentum=self.config.momentum, 
                                            nesterov=False 
                                        )
        elif self.config.optimizer_name == ADA_GRAD:
            print('\nUsing AdaGrad')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}\n')
            optimizer = torch.optim.Adagrad (
                                                [
                                                    {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                                    {'params': no_decay_params, 'weight_decay': 0.0}
                                                ],
                                                lr=lr_schedule[0]
                                            )
        elif self.config.optimizer_name == SGD_NESTEROV:
            print('\nUsing SGD Nesterov')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}\n')
            print(f'Momentum: {self.config.momentum}')
            optimizer = torch.optim.SGD(
                                            [
                                                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                                {'params': no_decay_params, 'weight_decay': 0.0}
                                            ],
                                            lr=lr_schedule[0],
                                            momentum=self.config.momentum,  
                                            nesterov=True  
                                        )
        elif self.config.optimizer_name == AMS_GRAD:
            print('\nUsing AMS Grad')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}\n')
            optimizer = torch.optim.Adam(
                                            [
                                                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                                {'params': no_decay_params, 'weight_decay': 0.0}
                                            ],
                                            lr=lr_schedule[0],
                                            betas=self.config.betas,
                                            eps=1e-8,
                                            amsgrad=True
                                        )
        elif self.config.optimizer_name == R_ADAM:
            print('\nUsing RAdam')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}\n')
            optimizer = torch.optim.RAdam(
                                        [
                                            {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                            {'params': no_decay_params, 'weight_decay': 0.0}
                                        ],
                                        lr=lr_schedule[0],
                                        betas=(0.9, 0.999),  
                                        eps=1e-8
                                    )
        else:
            raise ValueError("Wrong Optimizer Name!!")

        print("Configuration Complete\n")
        
        #====================================================================================================================

        st_epoch = 0

        if self.config.load_check_point:
            print("\nLoading Checkpoint")
            checkpoint = torch.load(self.config.checkpoint_path + self.config.checkpoint_name + '.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.all_train_losses = checkpoint['all_train_losses']
            self.train_losses = checkpoint['train_losses']

            self.all_train_accuracies = checkpoint['all_train_accuracies']
            self.train_accuracies = checkpoint['train_accuracies']

            self.val_losses = checkpoint['val_losses']
            self.val_accuracies = checkpoint['val_accuracies']
            
            print('\nLoaded Checkpoint: ' + self.config.checkpoint_path + self.config.checkpoint_name)
            
            st_epoch = checkpoint['epoch'] + 1

            print(f'Starting Epoch for Training: {st_epoch}')

            lr = optimizer.param_groups[0]['lr']
            print("Learning rate of loaded model:", lr)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[st_epoch * len(train_loader) / self.config.gradient_accum_iters]
            
            print("Setting learning rate to:", optimizer.param_groups[0]['lr'])

        #====================================================================================================================
 
        torch.cuda.empty_cache()
        print("Computing Started")

        for epoch in range(st_epoch, self.config.num_epochs):
            st_time = 0

            self.model.train()

            train_loss_epoch = 0.0
            correct = 0
            total = 0
            cumulative_loss = 0
            optimizer.zero_grad()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.config.gradient_accum_iters
                cumulative_loss += loss.detach()
                loss.backward()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx % self.config.gradient_accum_iters == 0) and (batch_idx != 0):
                    if self.config.max_norm != False:
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm)
                    else:
                        norm = 0
                        
                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss = cumulative_loss.item()
                    train_loss_epoch += train_loss
                
                    self.all_train_losses.append(train_loss)

                    cumulative_loss = 0

                    self.all_train_accuracies.append(100.* correct / total)

                    set_lr = lr_schedule[int(((epoch * len(train_loader)) + batch_idx) / self.config.gradient_accum_iters)]
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = set_lr
        
                    if self.config.device == torch.device('cuda'): 
                        torch.cuda.synchronize()

                    time_taken = time.time() - st_time
                    st_time = time.time()
                    print(f'Epoch {epoch:4d} | Loss: {train_loss:.5f} | Accuracy: {self.all_train_accuracies[-1]:.4f} | Norm: {norm:.4f} | Batch: ({batch_idx:4d}/{len(train_loader):4d}) | Time: {time_taken*1000:.2f}ms | LR: {set_lr:.3e}')

            avg_train_loss = train_loss_epoch / len(train_loader) * self.config.gradient_accum_iters
            print(f'\nAvg. Train Loss - Epoch: {epoch}, Average Training Loss: {avg_train_loss:.4f}')
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(100.* correct / total)

            if self.config.eval_val_set:
                print("Estimating Val Loss")
                val_loss, val_accuracy = self.estimate_val_loss()
                print(f'Validation at Epoch: {epoch} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%\n')
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

            if (epoch % self.config.checkpoint_save_epoch == 0) and (epoch != 0):
                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'all_train_losses' : self.all_train_losses,
                    'train_losses' : self.train_losses,
                    'all_train_accuracies' : self.all_train_accuracies,
                    'train_accuracies' : self.train_accuracies,
                    'val_losses' : self.val_losses,
                    'val_accuracies' : self.val_accuracies,
                    'train_config': self.config
                }, self.config.checkpoint_path + self.config.model_name + '-Epoch-' + str(epoch) + '-' + date_time + '.pth')

                print("Checkpoint Saved")
        
        print("Training Complete")