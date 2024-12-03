import torch
from TrainConfig import *

checkpoints_names = [
            'checkpoint_path_1',
            'checkpoint_path_2',
            'checkpoint_path_n',
]

all_train_losses = []
train_losses = []

all_train_accuracies = []
train_accuracies = []

val_losses = []
val_accuracies = []

all_train_loss_file = open("./All_Train_Loss.csv", "w")
train_loss_file = open("./Train_Loss.csv", "w")

all_train_accuracies_file = open("./All_Train_Accuracies.csv", "w")
train_accuracies_file = open("./Train_Accuracies.csv", "w")

val_loss_file = open("./Val_Loss.csv", "w")
val_accuracies_file = open("./Val_Accuracies.csv", "w")

for checkpoints_name in checkpoints_names:
    checkpoint = torch.load(checkpoints_name)

    model_name = checkpoint['train_config'].model_name if checkpoint['train_config'].model_name != None else 'None'
    optimizer_name = checkpoint['train_config'].optimizer_name if checkpoint['train_config'].optimizer_name != None else 'None'
    max_lr = checkpoint['train_config'].max_lr if checkpoint['train_config'].max_lr != None else 'None'
    min_lr = checkpoint['train_config'].min_lr if checkpoint['train_config'].min_lr != None else 'None'
    start_lr = checkpoint['train_config'].start_lr if checkpoint['train_config'].start_lr != None else 'None'
    lr_decr = checkpoint['train_config'].lr_decrement if checkpoint['train_config'].lr_decrement != None else 'None'
    lr_incr = checkpoint['train_config'].lr_increment if checkpoint['train_config'].lr_increment != None else 'None'
    warmup_iters = checkpoint['train_config'].warmup_iters if checkpoint['train_config'].warmup_iters != None else 'None'

    print(model_name)
    print(optimizer_name)
    print(f"max_lr: {max_lr}")
    print(f"min_lr: {min_lr}")
    print(f"start_lr: {start_lr}")
    print(f"lr_decrement: {lr_decr}")
    print(f"lr_increment: {lr_incr}")
    print(f"warmup_iters: {warmup_iters}")
    print('================================================================')

    all_train_losses.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['all_train_losses'])
    train_losses.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['train_losses'])
    
    all_train_accuracies.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['all_train_accuracies'])
    train_accuracies.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['train_accuracies'])

    val_losses.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['val_losses'])
    val_accuracies.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['val_accuracies'])

    print('Done')

    del checkpoint

all_train_losses = list(zip(*all_train_losses))
train_losses = list(zip(*train_losses))

all_train_accuracies = list(zip(*all_train_accuracies))
train_accuracies = list(zip(*train_accuracies))

val_losses = list(zip(*val_losses))
val_accuracies = list(zip(*val_accuracies))

for sub_list in all_train_losses:
    for item in sub_list:
        all_train_loss_file.write(str(item) + ',')
        
    all_train_loss_file.write('\n')

for sub_list in train_losses:
    for item in sub_list:
        train_loss_file.write(str(item) + ',')
        
    train_loss_file.write('\n')

for sub_list in all_train_accuracies:
    for item in sub_list:
        all_train_accuracies_file.write(str(item) + ',')

    all_train_accuracies_file.write('\n')

for sub_list in train_accuracies:
    for item in sub_list:
        train_accuracies_file.write(str(item) + ',')

    train_accuracies_file.write('\n')

for sub_list in val_losses:
    for item in sub_list:
        val_loss_file.write(str(item) + ',')

    val_loss_file.write('\n')

for sub_list in val_accuracies:
    for item in sub_list:
        val_accuracies_file.write(str(item) + ',')

    val_accuracies_file.write('\n')

all_train_loss_file.close()
train_loss_file.close()
all_train_accuracies_file.close()
train_accuracies_file.close()
val_loss_file.close()
val_accuracies_file.close()