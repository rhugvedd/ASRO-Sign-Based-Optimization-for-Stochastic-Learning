from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
import gc

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

all_tokens = []
end_of_text_token_id = tokenizer.eos_token_id
tot_tokens_processed = 0
shard = 0

no_of_toks_in_shard = 1e8

save_path = './Datasets/FineWeb-Edu/'

for batch in fw:
    text = batch['text']

    tokens = tokenizer.encode(text, add_special_tokens=False)
    all_tokens.extend(tokens)
    
    all_tokens.append(end_of_text_token_id)

    tot_tokens_processed = len(all_tokens)

    print(f"Tokens processed: {tot_tokens_processed}")

    if tot_tokens_processed > no_of_toks_in_shard:
        all_tokens_tensor = torch.tensor(all_tokens, dtype=torch.int32)

        tensor_file_path = save_path + f'Fine-Web-Edu-Sample-1BT-Shard-{shard}.pt'
        torch.save(all_tokens_tensor, tensor_file_path)

        print(all_tokens_tensor.shape)
        print(f'Tensor saved to {tensor_file_path}')

        all_tokens.clear()
        all_tokens = []
        gc.collect()
        tot_tokens_processed = 0
        shard += 1

if all_tokens:
    all_tokens_tensor = torch.tensor(all_tokens, dtype=torch.int32)
    tensor_file_path = save_path + f'Fine-Web-Edu-Sample-1BT-Shard-{shard}.pt'
    torch.save(all_tokens_tensor, tensor_file_path)
    print(all_tokens_tensor.shape)
    print(f'Tensor saved to {tensor_file_path}')
