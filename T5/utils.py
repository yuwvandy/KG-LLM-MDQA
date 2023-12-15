import random
import numpy as np
import torch
from dataset import dataset_process_inf
from torch.utils.data import DataLoader

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inf_encode(model, tokenizer, source_text, source_len, device):
    dataset = dataset_process_inf(source_text, tokenizer, source_len)
    
    params = {
        "batch_size": len(source_text),
        "shuffle": False,
        "num_workers": 8,
    }

    loader = DataLoader(dataset, **params)

    with torch.no_grad():
        for _, data in enumerate(loader):
            ids = data['source_ids'].to(device)
            mask = data['source_mask'].to(device)

            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=512, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    return preds


