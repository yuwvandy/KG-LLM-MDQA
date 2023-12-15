import torch.nn as nn
from transformers import AutoModel
import numpy as np
import torch

class Retriever(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps))

    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
        vector = self.project(cls_rep)

        return vector

    def forward(self, batch):
        q_emb = self.encode_seq(batch['q_enc_btz'], batch['q_mask'])

        c_emb = self.encode_seq(batch['c_enc_btz'], batch['c_mask'])

        n_emb = self.encode_seq(batch['n_enc_btz'], batch['n_mask'])


        return {'q_emb': q_emb, 'c_emb': c_emb, 'n_emb': n_emb}


class Retriever_inf(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps))
    
    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
        vector = self.project(cls_rep)

        return vector

    def forward(self, input_ids, attention_mask):
        # try:
        #     print(11)
        #     emb = self.encode_seq(input_ids, attention_mask)
        # except:
        #     #split batch into three pieces and combine, using len//6
        #     emb = torch.cat([self.encode_seq(input_ids[i:i+len(input_ids)//6], attention_mask[i:i+len(input_ids)//6]) for i in range(0, len(input_ids), len(input_ids)//6)], dim = 0)

        #split batch into two pieces
        # emb_1 = self.encode_seq(input_ids[:input_ids.shape[0]//2], attention_mask[:input_ids.shape[0]//2])
        # emb_2 = self.encode_seq(input_ids[input_ids.shape[0]//2:], attention_mask[input_ids.shape[0]//2:])

        #split batch int four pieces
        emb_1 = self.encode_seq(input_ids[:input_ids.shape[0]//4], attention_mask[:input_ids.shape[0]//4])
        emb_2 = self.encode_seq(input_ids[input_ids.shape[0]//4:input_ids.shape[0]//2], attention_mask[input_ids.shape[0]//4:input_ids.shape[0]//2])
        emb_3 = self.encode_seq(input_ids[input_ids.shape[0]//2:input_ids.shape[0]//4*3], attention_mask[input_ids.shape[0]//2:input_ids.shape[0]//4*3])
        emb_4 = self.encode_seq(input_ids[input_ids.shape[0]//4*3:], attention_mask[input_ids.shape[0]//4*3:])

        return torch.cat([emb_1, emb_2, emb_3, emb_4], axis = 0)
