import torch

checkpoints_names = [
            'checkpoint_path_1',
            'checkpoint_path_2',
            'checkpoint_path_n',
]

all_tot_loss = []
all_val_loss = []
all_norms = []

tot_loss_file = open("./Total_Loss.csv", "w")
val_loss_file = open("./Val_Loss.csv", "w")
norm_file = open("./Norms.csv", "w")

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

    all_tot_loss.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['total_loss_list'])
    all_val_loss.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['val_losses'])
    all_norms.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(start_lr) + '-' + str(lr_decr) + '-' + str(lr_incr)] + checkpoint['total_norm_list'])

    print('Done')

    del checkpoint

all_tot_loss = list(zip(*all_tot_loss))
all_val_loss = list(zip(*all_val_loss))
all_norms = list(zip(*all_norms))

for sub_list in all_tot_loss:
    for item in sub_list:
        tot_loss_file.write(str(item) + ',')
        
    tot_loss_file.write('\n')

for sub_list in all_val_loss:
    for item in sub_list:
        val_loss_file.write(str(item) + ',')
        
    val_loss_file.write('\n')

for sub_list in all_norms:
    for item in sub_list:
        norm_file.write(str(item) + ',')

    norm_file.write('\n')

tot_loss_file.close()
val_loss_file.close()
norm_file.close()