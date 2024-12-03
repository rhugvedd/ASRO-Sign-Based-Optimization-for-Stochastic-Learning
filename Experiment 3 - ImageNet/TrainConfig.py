import torch

class TrainConfig:
    def __init__    (   
                        self, 
                        batch_size: int,
                        max_norm: float,
                        gradient_accum_iters: int,
                        print_stat_itrvl: int,
                        device: torch.device,
                        dropout: float,
                        weight_decay: float,
                        load_check_point: bool,
                        checkpoint_path: str,
                        checkpoint_name: str,
                        checkpoint_save_epoch: int,
                        num_epochs: int,
                        eval_val_set: bool,
                        val_eval_iters: int,
                        val_eval_interval: int,
                        optimizer_name: str,
                        betas: tuple,
                        max_lr: float,
                        min_lr: float,
                        model_name: str,
                        start_lr: float,
                        lr_decrement: float,
                        lr_increment: float,
                        warmup_iters: int,
                        data_path: str,
                        alpha = None,
                        momentum = None
                    ):

        self.batch_size = batch_size
        self.max_norm = max_norm
        self.gradient_accum_iters = gradient_accum_iters
        self.print_stat_itrvl = print_stat_itrvl
        self.device = device
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.load_check_point = load_check_point
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.checkpoint_save_epoch = checkpoint_save_epoch
        self.num_epochs = num_epochs
        self.eval_val_set = eval_val_set
        self.val_eval_iters = val_eval_iters
        self.val_eval_interval = val_eval_interval
        self.optimizer_name = optimizer_name
        self.betas = betas
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.model_name = model_name
        self.start_lr = start_lr
        self.lr_decrement = lr_decrement
        self.lr_increment = lr_increment
        self.warmup_iters = warmup_iters
        self.data_path = data_path
        self.alpha = alpha
        self.momentum = momentum