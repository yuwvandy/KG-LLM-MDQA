
import numpy as np
import torch.nn as nn
from transformers import AutoModel
import numpy as np
import torch



class Retriever_inf(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.mhop_model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps))
    
    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
        vector = self.project(cls_rep)

        return vector

    def forward(self, input_ids, attention_mask):
        #split batch into two pieces
        emb_1 = self.encode_seq(input_ids[:input_ids.shape[0]//2], attention_mask[:input_ids.shape[0]//2])
        emb_2 = self.encode_seq(input_ids[input_ids.shape[0]//2:], attention_mask[input_ids.shape[0]//2:])

        return torch.cat([emb_1, emb_2], axis = 0)

@torch.no_grad()
def run(d, model, tokenizer, args):
    model.eval()
    
    titles = [title for title, _ in d['title_chunks']]
    chunks = [chunk for _, chunk in d['title_chunks']]

    chunks_encode = tokenizer(text = titles, text_pair = chunks, max_length = args.max_len, return_tensors = 'pt', padding=True, truncation=True)

    for key in chunks_encode:
        chunks_encode[key] = chunks_encode[key].to(args.device)

    c_emb = model(chunks_encode['input_ids'], chunks_encode['attention_mask'])

    return c_emb.cpu().numpy()