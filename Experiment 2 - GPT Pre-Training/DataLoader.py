import torch
from datetime import datetime
import gc

class DataLoader:
    def __init__    (
                        self,
                        save_path: str
                    ):
        super(DataLoader, self).__init__()

        self.save_path = save_path
        self.train_batch_index = 0
        self.val_batch_index = 0
        
    def shuffle (
                    self,
                    split: str,
                    reset_batch_index: bool
                ):
        
        if split == 'train':
            print("SHUFFLING TRAIN BATCHES")
            shuffle_indices = torch.randperm(self.x_train.size(0))
            self.x_train = self.x_train.index_select(0, shuffle_indices)
            self.y_train = self.y_train.index_select(0, shuffle_indices)

            if reset_batch_index:
                self.train_batch_index = 0

        elif split == 'val':
            print("SHUFFLING VAL BATCHES")
            shuffle_indices = torch.randperm(self.x_val.size(0))
            self.x_val = self.x_val.index_select(0, shuffle_indices)
            self.y_val = self.y_val.index_select(0, shuffle_indices)
            
            if reset_batch_index:
                self.val_batch_index = 0
        else:
            raise ValueError("Wrong split name!")

    def load_shard  (
                        self,
                        shard_name: str,
                        train_val: str
                    ):
        batch_toks = self.batch_size * self.context_size

        if not (self.batch_overlap >= 0 and self.batch_overlap <= self.context_size and (self.batch_overlap % 1) == 0):
            raise ValueError("'batch_overlap' must be between 0 and 'context_size'") 

        data = torch.load(self.save_path + shard_name + '.pt')

        x_data = data[:-1]
        y_data = data[1:]

        batch_non_overlap = self.context_size - self.batch_overlap

        print([(batch_jump + batch_st, batch_jump + batch_st + batch_toks) for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, self.context_size, batch_non_overlap)][-1])

        x_data = torch.stack([x_data[batch_jump + batch_st : batch_jump + batch_st + batch_toks] for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, self.context_size, batch_non_overlap)], dim = 0)
        y_data = torch.stack([y_data[batch_jump + batch_st : batch_jump + batch_st + batch_toks] for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, self.context_size, batch_non_overlap)], dim = 0)
        
        x_data = x_data.view(-1, self.batch_size, self.context_size)
        y_data = y_data.view(-1, self.batch_size, self.context_size)

        x_data = x_data.to(self.x_dtype)
        y_data = y_data.to(self.y_dtype)

        num_batches = x_data.size(0)

        if train_val == 'train':
            self.x_train = x_data
            self.y_train = y_data
            self.train_num_batches = num_batches
            print(f"Loaded 'Train' Shard - {shard_name}")
        elif train_val == 'val':
            self.x_val = x_data
            self.y_val = y_data
            self.val_num_batches = num_batches
            print(f"Loaded 'Val' Shard - {shard_name}")
        else:
            raise ValueError('Wrong split name! Expected "train" or "val".')

        gc.collect()

    def load_data   (
                        self, 
                        batch_size: int,
                        context_size: int,
                        train_shard_names: list,
                        batch_overlap: float, # Should be between 0 and context size.
                        x_dtype: torch.dtype,
                        y_dtype: torch.dtype,
                        val_name = None,
                        load_shard_indx = 0,
                        load_train_batch_indx = 0
                    ):

        self.batch_size = batch_size
        self.context_size = context_size
        self.train_shard_names = train_shard_names
        self.batch_overlap = batch_overlap
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

        self.shard_indx = load_shard_indx
        self.tot_shards = len(train_shard_names)

        self.load_shard(train_shard_names[self.shard_indx], train_val = 'train')
        
        self.train_batch_index = load_train_batch_indx
        #print('Lswoifiscoicncke cewiknceocwenoweinowe')
        #print(self.train_batch_indx)
        print(f'Batch Index of First Batch to load: {self.train_batch_index}')

        if val_name != None:
            self.load_shard(val_name, train_val = 'val')

    def get_train_batch (
                            self, 
                            device: torch.device
                        ):
        batch_x = self.x_train[self.train_batch_index].to(device)
        batch_y = self.y_train[self.train_batch_index].to(device)

        self.train_batch_index = (self.train_batch_index + 1) % self.train_num_batches

        if self.train_batch_index == (self.train_num_batches - 1):
            print(f"\nLoaded last batch of the Shard - {self.train_shard_names[self.shard_indx]}\n")

            self.train_batch_index = 0
            self.shard_indx = (self.shard_indx + 1) % self.tot_shards
            

            self.load_shard(self.train_shard_names[self.shard_indx], train_val = 'train')

        return batch_x, batch_y

    def get_val_batch   (
                            self,
                            device: torch.device
                        ):
        
        batch_x = self.x_val[self.val_batch_index].to(device)
        batch_y = self.y_val[self.val_batch_index].to(device)

        self.val_batch_index = (self.val_batch_index + 1) % self.val_num_batches
        
        return batch_x, batch_y
    
    def set_indx (
                    self,
                    batch_index: int,
                    train_val: str
                 ):

        if train_val == 'train':
            self.train_batch_index = batch_index
        elif train_val == 'val':
            self.val_batch_index = batch_index
        else:
            raise ValueError('Wrong split name! Expected "train" or "val".')

    def reset   (
                    self,
                    train_or_val: str
                ):

        if train_or_val == 'train':
            self.train_batch_index = 0
        elif train_or_val == 'val':
            self.val_batch_index = 0
        else:
            raise ValueError('Wrong split name! Expected "train" or "val".')